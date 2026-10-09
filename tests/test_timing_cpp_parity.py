"""C++ timing runtime vs Python: quantiles, offsets (median and seeded takes) and written MIDI must match.

Requires the built CLI (cpp/build) and models/timing_xgb/model.hbm (export_cpp_model.py):

    python tests/test_timing_cpp_parity.py                    # synthetic + quantized MAESTRO pieces
    python tests/test_timing_cpp_parity.py --maestro_files 6
    python -m pytest tests/test_timing_cpp_parity.py         # skipped if the CLI or the model is missing

Python runs humanize_timing with ``--beat_source midi`` (the grid from the file's own tempo map),
the only grid the C++ runtime implements and the path quantized DAW input takes. Inputs are
synthetic scores (tempo and time-signature changes, triplets, chords, re-strikes, drums, pedal)
and quantized versions of MAESTRO test pieces: their reconstructed scores written as MIDI.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import mido
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humanize_timing import humanize  # noqa: E402
from midi_io import load_midi  # noqa: E402
from model_io import TimingModel  # noqa: E402
from test_cpp_parity import CLI, parse_order, synthetic_score  # noqa: E402
from timing_features import featurize_quantized, reconstruct_score  # noqa: E402
from utils import load_features_meta, setup_logging  # noqa: E402

MODEL = ROOT / "models/timing_xgb/model.json"
CASES = {  # name: (C++ CLI options, Python humanize_timing arguments)
    "median": (["--timing-mode", "median"], {"mode": "median"}),
    "seed 7": (["--seed", "7"], {"seed": 7}),
    "seed 42, T 0.7, x1.3, overrides": (
        ["--seed", "42", "--temperature", "0.7", "--timing-scale", "1.3", "--chord-coupling", "0.9",
         "--correlation-beats", "2"],
        {"seed": 42, "temperature": 0.7, "scale": 1.3, "chord_coupling": 0.9, "correlation_beats": 2.0}),
    "snap, seed 3": (["--seed", "3", "--snap"], {"seed": 3, "snap": True}),
}


def quantized_maestro(src: Path, dst: Path, model: TimingModel) -> Path:
    """A MAESTRO performance's reconstructed score as DAW-style MIDI (constant tempo, real ticks)."""
    md = load_midi(src)
    sc = reconstruct_score(md, model.pipeline_config())
    tpb = 480
    on = np.rint(sc.onset_beat * tpb).astype(int)
    off = np.maximum(np.rint(sc.offset_beat * tpb).astype(int), on + 1)
    bpm = float(np.median(sc.score_grid.tempo_bpm))
    meter = int(np.bincount(sc.raw_grid.beats_per_bar).argmax())
    events = []
    for k in range(len(md.notes)):
        p, v = int(md.notes.pitch[k]), int(md.notes.velocity[k])
        events += [(on[k], 1, mido.Message("note_on", note=p, velocity=v)), (off[k], 0, mido.Message("note_off", note=p))]
    track, last = mido.MidiTrack(), 0
    for t, _, m in sorted(events, key=lambda e: (e[0], e[1])):
        track.append(m.copy(time=int(t - last)))
        last = t
    meta = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0),
                           mido.MetaMessage("time_signature", numerator=meter, denominator=4, time=0)])
    mf = mido.MidiFile(type=1, ticks_per_beat=tpb)
    mf.tracks += [meta, track]
    mf.save(str(dst))
    return dst


def events(path: Path) -> list:
    """Every message of every track with its absolute tick."""
    out = []
    for tr in mido.MidiFile(str(path), clip=True).tracks:
        t, rows = 0, []
        for m in tr:
            t += m.time
            rows.append((t, m.copy(time=0)))
        out.append(rows)
    return out


def compare_file(model: TimingModel, hbm: Path, midi_path: Path, tmp: Path) -> list[dict]:
    import subprocess

    md = load_midi(midi_path)
    to_parse = parse_order(md)
    rows = []
    for name, (cli_args, py_args) in CASES.items():
        snap = bool(py_args.get("snap", False))
        piece = featurize_quantized(md, model.pipeline_config(), "midi", snap=snap)
        q_py = model.predict_quantiles(piece.df)
        py_out, cpp_out, dump = tmp / "py.mid", tmp / "cpp.mid", tmp / "timing.csv"
        ns = argparse.Namespace(input=str(midi_path), model=str(model.path), output=str(py_out), mode="sample",
                                temperature=1.0, scale=1.0, max_offset_ms=150.0, seed=None, chord_coupling=None,
                                correlation_beats=None, beat_source="midi", snap=False)
        for k, v in py_args.items():
            setattr(ns, k, v)
        ms_py = humanize(ns)
        subprocess.run([str(CLI), "--timing-model", str(hbm), "--input", str(midi_path), "--output", str(cpp_out),
                        "--dump-timing", str(dump), *cli_args], check=True, capture_output=True)
        c = pd.read_csv(dump, float_precision="round_trip")
        k = q_py.shape[1]
        q_cpp = c[[f"q{j}" for j in range(k)]].to_numpy()
        rows.append({
            "file": midi_path.name[:40], "case": name, "notes": len(md.notes),
            "row_order_ok": bool(np.array_equal(c["note_index"].to_numpy(), to_parse[piece.df["note_order"].to_numpy()])),
            "max_quantile_diff_ms": float(np.max(np.abs(q_cpp - q_py))),
            "max_offset_diff_ms": float(np.max(np.abs(c["offset_ms"].to_numpy() - ms_py))),
            "midi_events_identical": events(cpp_out) == events(py_out),
            "offset_sd_ms": float(np.std(ms_py)),
        })
    return rows


def run(maestro_files: int = 3) -> pd.DataFrame:
    model = TimingModel.load(MODEL)
    hbm = MODEL.with_suffix(".hbm")
    rows = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        inputs = [synthetic_score(tmp / f"synthetic_{s}.mid", s) for s in range(3)]
        fmeta = load_features_meta(ROOT / "data/features.parquet")
        test_files = sorted(f["midi_filename"] for f in fmeta["files"] if f["split"] == "test")
        rng = np.random.default_rng(1)
        for i, name in enumerate(rng.choice(test_files, size=min(maestro_files, len(test_files)), replace=False)):
            inputs.append(quantized_maestro(Path(fmeta["maestro_dir"]) / name, tmp / f"maestro_quantized_{i}.mid", model))
        for path in inputs:
            rows += compare_file(model, hbm, path, tmp)
    return pd.DataFrame(rows)


def ok(df: pd.DataFrame) -> bool:
    return bool(df["row_order_ok"].all() and (df["max_quantile_diff_ms"] < 1e-4).all()
                and (df["max_offset_diff_ms"] < 1e-6).all() and df["midi_events_identical"].all())


def test_timing_cpp_parity() -> None:
    import pytest

    if CLI is None or not MODEL.with_suffix(".hbm").exists():
        pytest.skip("C++ CLI or exported timing model not found")
    df = run(maestro_files=1)
    assert ok(df), df.to_string()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro_files", type=int, default=3)
    args = ap.parse_args()
    setup_logging("ERROR")
    if CLI is None:
        sys.exit("build the C++ CLI first (see cpp/README.md)")
    result = run(args.maestro_files)
    pd.set_option("display.width", 250)
    print(result.to_string(index=False))
    print("\nPARITY OK" if ok(result) else "\nPARITY MISMATCH")
    sys.exit(0 if ok(result) else 1)
