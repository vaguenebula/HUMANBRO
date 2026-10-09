"""C++ runtime vs Python pipeline: features, raw predictions and written velocities must match.

Requires the built CLI (cpp/build/humanbro[.exe]) and exported models (export_cpp_model.py):

    python tests/test_cpp_parity.py                       # quantized + primary model, synthetic + MAESTRO files
    python tests/test_cpp_parity.py --maestro_files 10
    python -m pytest tests/test_cpp_parity.py            # skipped if the CLI or models are missing

The comparison uses Python with ``--beat_source midi`` (the grid from the file's own tempo
map and time signatures), which is the only beat source the C++ runtime implements.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
import tempfile
from pathlib import Path

import mido
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humanize_midi import humanize  # noqa: E402
from midi_features import featurize_midi  # noqa: E402
from midi_io import load_midi  # noqa: E402
from model_io import VelocityModel  # noqa: E402
from utils import load_features_meta, setup_logging  # noqa: E402

CLI = next((p for p in (ROOT / "cpp/build/humanbro.exe", ROOT / "cpp/build/humanbro", ROOT / "cpp/build/Release/humanbro.exe") if p.exists()), None)
MODELS = [ROOT / "models/velocity_xgb_quantized/model.json", ROOT / "models/velocity_xgb/model.json"]
OPTION_SETS = {
    "default": [],
    "smooth+shape": ["--smoothing", "--dynamics-scale", "1.2", "--offset", "-3"],
}


def synthetic_score(path: Path, seed: int) -> Path:
    """Grid-aligned piece with tempo and time-signature changes, chords, re-strikes, drums."""
    rng = np.random.default_rng(seed)
    tpb = 480
    meta = [(0, mido.MetaMessage("time_signature", numerator=4, denominator=4))]
    events = []
    tick = 0
    for bar, (num, den) in enumerate([(4, 4)] * 4 + [(3, 4)] * 4 + [(6, 8)] * 4 + [(4, 4)] * 4):
        if bar in (4, 8, 12):
            meta.append((tick, mido.MetaMessage("time_signature", numerator=num, denominator=den)))
        if bar % 3 == 0:
            meta.append((tick, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(rng.uniform(60, 160))))))
        unit = tpb * 4 // den
        for beat in range(num):
            t = tick + beat * unit
            chord = rng.choice(np.arange(48, 72), size=int(rng.integers(1, 5)), replace=False)
            for pch in chord:
                dur = int(unit * rng.choice([0.5, 1, 2]))
                events.append((t, mido.Message("note_on", note=int(pch), velocity=int(rng.integers(20, 110)))))
                events.append((t + dur, mido.Message("note_off", note=int(pch))))
            for sub in range(int(rng.integers(1, 4))):
                tm = t + sub * unit // 3
                pch = int(rng.integers(72, 90))
                events.append((tm, mido.Message("note_on", note=pch, velocity=int(rng.integers(20, 110)))))
                events.append((tm + unit // 4, mido.Message("note_on", note=pch, velocity=0)))
            events.append((t, mido.Message("note_on", channel=9, note=36, velocity=100)))  # drums: ignored
            events.append((t + 10, mido.Message("note_off", channel=9, note=36)))
            events.append((t + unit // 2, mido.Message("control_change", control=64, value=int(rng.integers(0, 128)))))
        tick += num * unit
    # A re-strike of a held key and a note that is never released.
    events += [(tick, mido.Message("note_on", note=60, velocity=50)), (tick + 100, mido.Message("note_on", note=60, velocity=70)),
               (tick + 300, mido.Message("note_off", note=60)), (tick + 400, mido.Message("note_on", note=40, velocity=80))]

    def track(evts: list) -> mido.MidiTrack:
        tr, last = mido.MidiTrack(), 0
        for t, m in sorted(evts, key=lambda e: e[0]):
            tr.append(m.copy(time=t - last))
            last = t
        return tr

    mf = mido.MidiFile(type=1, ticks_per_beat=tpb)
    mf.tracks += [track(meta), track(events)]
    mf.save(str(path))
    return path


def parse_order(md) -> np.ndarray:
    """For each NoteArray row, its index in file (parse) order = the C++ Score index."""
    rank = np.lexsort((md.notes.msg_index, md.notes.track))
    out = np.empty(len(rank), dtype=np.int64)
    out[rank] = np.arange(len(rank))
    return out


def compare_file(model: VelocityModel, hbm: Path, midi_path: Path, tmp: Path) -> dict:
    md = load_midi(midi_path)
    cfg = model.pipeline_config()
    cfg = dataclasses.replace(cfg, beat=dataclasses.replace(cfg.beat, source="midi"))
    df = featurize_midi(md, cfg).df
    py_feat = df[model.feature_columns].to_numpy(np.float32)
    py_raw = model.predict_raw(df)
    to_parse = parse_order(md)

    feat_csv, raw_csv, out_mid = tmp / "f.csv", tmp / "r.csv", tmp / "o.mid"
    subprocess.run([str(CLI), "--model", str(hbm), "--input", str(midi_path), "--output", str(out_mid),
                    "--dump-features", str(feat_csv), "--dump-raw", str(raw_csv)], check=True, capture_output=True)
    cf = pd.read_csv(feat_csv)
    cr = pd.read_csv(raw_csv)
    res = {"file": midi_path.name[:48], "notes": len(df)}
    res["row_order_ok"] = bool(np.array_equal(cf["note_index"].to_numpy(), to_parse[df["note_order"].to_numpy()]))
    c_feat = cf[model.feature_columns].to_numpy(np.float32)
    same = (c_feat == py_feat) | (np.isnan(c_feat) & np.isnan(py_feat))
    res["feature_values_differing"] = int((~same).sum())
    bad_cols = [model.feature_columns[j] for j in np.flatnonzero((~same).any(axis=0))]
    res["columns_differing"] = ",".join(bad_cols[:4]) + ("..." if len(bad_cols) > 4 else "")
    res["max_raw_diff"] = float(np.max(np.abs(cr["raw"].to_numpy() - py_raw)))

    mismatched = 0
    for name, extra in OPTION_SETS.items():
        cpp_out, py_out = tmp / f"cpp_{name}.mid", tmp / f"py_{name}.mid"
        subprocess.run([str(CLI), "--model", str(hbm), "--input", str(midi_path), "--output", str(cpp_out), *extra],
                       check=True, capture_output=True)
        ns = argparse.Namespace(model=str(model.path), input=str(midi_path), output=str(py_out), beat_source="midi",
                                fixed_meter=None, smoothing="--smoothing" in extra, smoothing_alpha=0.35,
                                smoothing_strength=0.5, accent_threshold=8.0,
                                dynamics_scale=1.2 if extra else 1.0, offset=-3.0 if extra else 0.0, mix=1.0,
                                baseline="input", base_velocity=64.0)
        humanize(ns)
        a, b = load_midi(cpp_out), load_midi(py_out)
        assert np.array_equal(a.notes.onset_tick, b.notes.onset_tick) and np.array_equal(a.notes.pitch, b.notes.pitch)
        mismatched += int(np.sum(a.notes.velocity != b.notes.velocity))
    res["velocities_differing"] = mismatched
    # Every event except note-on velocities must be unchanged (pedals, tempo, note-offs, drums, ...).
    def events(path: Path) -> list:
        mf = mido.MidiFile(str(path), clip=True)
        strip = lambda m: m.copy(velocity=1) if m.type == "note_on" and m.velocity > 0 and m.channel != 9 else m  # noqa: E731
        return [[strip(m) for m in tr] for tr in mf.tracks]

    src, out = Path(midi_path).read_bytes(), (tmp / "cpp_default.mid").read_bytes()
    res["other_events_preserved"] = len(src) == len(out) and events(Path(midi_path)) == events(tmp / "cpp_default.mid")
    return res


def run(maestro_files: int = 4) -> pd.DataFrame:
    rows = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        inputs = [synthetic_score(tmp / f"synthetic_{s}.mid", s) for s in range(3)]
        fmeta = load_features_meta(ROOT / "data/features.parquet")
        test_files = sorted(f["midi_filename"] for f in fmeta["files"] if f["split"] == "test")
        rng = np.random.default_rng(0)
        for name in rng.choice(test_files, size=min(maestro_files, len(test_files)), replace=False):
            inputs.append(Path(fmeta["maestro_dir"]) / name)
        for model_json in MODELS:
            hbm = model_json.with_suffix(".hbm")
            if not hbm.exists():
                continue
            model = VelocityModel.load(model_json)
            for path in inputs:
                rows.append({"model": model_json.parent.name, **compare_file(model, hbm, path, tmp)})
    return pd.DataFrame(rows)


def test_cpp_parity() -> None:
    import pytest

    if CLI is None or not any(m.with_suffix(".hbm").exists() for m in MODELS):
        pytest.skip("C++ CLI or exported models not found")
    df = run(maestro_files=2)
    assert df["row_order_ok"].all()
    assert (df["feature_values_differing"] == 0).all(), df.to_string()
    assert (df["max_raw_diff"] < 1e-4).all()
    assert (df["velocities_differing"] == 0).all()
    assert df["other_events_preserved"].all()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro_files", type=int, default=4)
    args = ap.parse_args()
    setup_logging("ERROR")
    if CLI is None:
        sys.exit("build the C++ CLI first (see cpp/README.md)")
    result = run(args.maestro_files)
    pd.set_option("display.width", 250)
    print(result.to_string(index=False))
    ok = (result["row_order_ok"].all() and (result["feature_values_differing"] == 0).all()
          and (result["velocities_differing"] == 0).all() and result["other_events_preserved"].all())
    print("\nPARITY OK" if ok else "\nPARITY MISMATCH")
    sys.exit(0 if ok else 1)
