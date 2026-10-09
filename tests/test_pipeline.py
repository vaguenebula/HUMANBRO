"""Self-contained checks for the parts most likely to go silently wrong.

Run with ``python -m pytest tests`` or ``python tests/test_pipeline.py``.
No dataset needed: synthetic MIDI files with known beats are generated here.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beat_tracking import build_beat_grid, grid_from_midi  # noqa: E402
from config import Config, load_config  # noqa: E402
from midi_features import (  # noqa: E402
    assert_no_leakage,
    canonical_velocities,
    compute_targets,
    featurize_midi,
    residual_baseline,
)
from midi_io import load_midi, looks_quantized, parse_midi, with_velocities  # noqa: E402
from postprocess import smooth_velocities  # noqa: E402

TPB = 480


def synthetic_piece(meter: int, bars: int, bpm_curve: list[float], jitter_s: float = 0.0, seed: int = 0) -> mido.MidiFile:
    """Bass on beat 1, chords on the other beats, a melody in eighths; one tempo event per bar."""
    rng = np.random.default_rng(seed)
    events: list[tuple[int, mido.Message]] = []
    meta: list[tuple[int, mido.MetaMessage]] = [(0, mido.MetaMessage("time_signature", numerator=meter, denominator=4))]
    melody = [72, 74, 76, 77, 79, 77, 76, 74]
    for bar in range(bars):
        bar_tick = bar * meter * TPB
        bpm = bpm_curve[min(bar, len(bpm_curve) - 1)]
        meta.append((bar_tick, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
        for beat in range(meter):
            t = bar_tick + beat * TPB
            notes = [(36 + (bar % 4) * 2, TPB * meter - 10)] if beat == 0 else [(55, TPB // 2), (60, TPB // 2), (64, TPB // 2)]
            for pitch, dur in notes:
                events.append((t, mido.Message("note_on", note=pitch, velocity=int(rng.integers(30, 100)))))
                events.append((t + dur, mido.Message("note_on", note=pitch, velocity=0)))
            for half in range(2):
                tm = t + half * TPB // 2
                pitch = melody[(bar * meter * 2 + beat * 2 + half) % len(melody)]
                events.append((tm, mido.Message("note_on", note=pitch, velocity=int(rng.integers(30, 100)))))
                events.append((tm + TPB // 2 - 20, mido.Message("note_on", note=pitch, velocity=0)))
        events.append((bar_tick + meter * TPB - 1, mido.Message("control_change", control=64, value=0)))

    def to_track(evts: list) -> mido.MidiTrack:
        tr = mido.MidiTrack()
        last = 0
        # note-offs (velocity 0) before note-ons at the same tick
        for tick, msg in sorted(evts, key=lambda e: (e[0], getattr(e[1], "velocity", 1) > 0)):
            tr.append(msg.copy(time=tick - last))
            last = tick
        return tr

    mf = mido.MidiFile(type=1, ticks_per_beat=TPB)
    mf.tracks.append(to_track(meta))
    mf.tracks.append(to_track(events))
    if jitter_s > 0:  # turn it into a "performance": re-time with a dummy 120 BPM map + jitter
        md = parse_midi(mf)
        perf = mido.MidiFile(type=1, ticks_per_beat=TPB)
        perf.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000, time=0)]))
        evts = []
        for k in range(len(md.notes)):
            j = rng.normal(0, jitter_s)
            on = int(round((md.notes.onset[k] + j) * 2 * TPB))
            off = int(round((md.notes.offset[k] + j) * 2 * TPB))
            evts.append((max(on, 0), mido.Message("note_on", note=int(md.notes.pitch[k]), velocity=int(md.notes.velocity[k]))))
            evts.append((max(off, on + 1), mido.Message("note_on", note=int(md.notes.pitch[k]), velocity=0)))
        perf.tracks.append(to_track(evts))
        return perf
    return mf


def true_beats(meter: int, bars: int, bpm_curve: list[float]) -> np.ndarray:
    t, out = 0.0, []
    for bar in range(bars):
        ibi = 60.0 / bpm_curve[min(bar, len(bpm_curve) - 1)]
        for _ in range(meter):
            out.append(t)
            t += ibi
    return np.array(out)


def beat_f_measure(est: np.ndarray, ref: np.ndarray, tol: float = 0.07) -> float:
    est = est[(est >= ref[0] - tol) & (est <= ref[-1] + tol)]
    hits = sum(np.min(np.abs(est - r)) <= tol for r in ref) if len(est) else 0
    prec = hits / max(len(est), 1)
    rec = hits / len(ref)
    return 0.0 if hits == 0 else 2 * prec * rec / (prec + rec)


def _save_load(mf: mido.MidiFile):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.mid"
        mf.save(str(path))
        return load_midi(path)


def test_tempo_map_and_midi_grid() -> None:
    curve = [100, 100, 80, 80, 140, 140]
    md = _save_load(synthetic_piece(3, 6, curve))
    grid = grid_from_midi(md)
    ref = true_beats(3, 6, curve)
    assert np.allclose(grid.times[: len(ref)], ref, atol=1e-6), "tick->seconds must follow every tempo change"
    assert grid.beats_per_bar[0] == 3 and grid.beat_in_bar[3] == 0
    assert looks_quantized(md)


def test_tracked_beats_and_meter() -> None:
    cfg = Config()
    for meter in (3, 4):
        curve = list(np.linspace(90, 120, 24))
        md = _save_load(synthetic_piece(meter, 24, curve, jitter_s=0.012, seed=meter))
        assert not looks_quantized(md)
        grid = build_beat_grid(md, cfg.beat)
        f = beat_f_measure(grid.times, true_beats(meter, 24, curve))
        assert f > 0.9, f"beat F-measure {f:.2f} for {meter}/4"
        assert grid.beats_per_bar[len(grid.times) // 2] == meter, f"meter estimate {grid.meter_label} for {meter}/4"


def test_features_ignore_velocity() -> None:
    md = _save_load(synthetic_piece(4, 8, [110], jitter_s=0.01))
    cfg = Config()
    a = featurize_midi(md, cfg).df
    flat = parse_midi(with_velocities(md, np.full(len(md.notes), 64)), md.path)
    rnd = parse_midi(with_velocities(md, np.random.default_rng(3).integers(1, 128, len(md.notes))), md.path)
    assert a.equals(featurize_midi(flat, cfg).df)
    assert a.equals(featurize_midi(rnd, cfg).df)


def test_velocity_write_roundtrip_preserves_other_events() -> None:
    md = _save_load(synthetic_piece(4, 4, [120]))
    new_v = np.arange(len(md.notes)) % 127 + 1
    out = parse_midi(with_velocities(md, new_v), md.path)
    assert np.array_equal(out.notes.velocity, new_v)
    assert np.array_equal(out.notes.onset_tick, md.notes.onset_tick)
    assert np.array_equal(out.notes.offset_tick, md.notes.offset_tick)
    cc = lambda m: sum(msg.type == "control_change" for tr in m.midi.tracks for msg in tr)  # noqa: E731
    assert cc(out) == cc(md)


def test_residual_baseline_excludes_own_chord() -> None:
    v = np.array([10, 10, 100, 100, 10, 10], dtype=float)
    g = np.array([0, 1, 2, 2, 3, 4])
    base = residual_baseline(v, g, window=2)
    assert base[2] == 10 and base[3] == 10  # the loud chord does not see itself


def test_targets_and_leakage_guard() -> None:
    md = _save_load(synthetic_piece(4, 4, [120], jitter_s=0.01))
    cfg = Config()
    pf = featurize_midi(md, cfg)
    t = compute_targets(canonical_velocities(md, pf.df), pf.df["group_id"].to_numpy(), cfg.target)
    assert np.allclose(t["velocity"] - t["baseline_velocity"], t["velocity_delta"], atol=1e-4)
    assert_no_leakage(list(pf.groups))
    for bad in (["pitch", "velocity"], ["pc_prev_group_mean_velocity"], ["baseline_velocity"]):
        try:
            assert_no_leakage(bad)
        except ValueError:
            continue
        raise AssertionError(f"leakage guard accepted {bad}")


def test_smoothing_preserves_voicing_and_accents() -> None:
    v = np.array([60, 70, 62, 72, 100, 110, 61, 71], dtype=float)
    g = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    s = smooth_velocities(v, g, alpha=0.5, strength=1.0, accent_threshold=8.0)
    assert np.allclose(np.diff(s.reshape(4, 2), axis=1), 10)  # intra-chord voicing kept
    assert np.allclose(s[4:6], v[4:6])  # accent group untouched


def test_config_yaml_matches_defaults() -> None:
    yaml_path = Path(__file__).resolve().parents[1] / "config.yaml"
    assert load_config(yaml_path).to_dict() == Config().to_dict()


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # report all failures, not just the first
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
