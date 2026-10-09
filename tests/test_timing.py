"""Checks for the timing pipeline (no dataset needed).

Run with ``python -m pytest tests`` or ``python tests/test_timing.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config  # noqa: E402
from midi_io import parse_midi, with_timing  # noqa: E402
from test_pipeline import TPB, synthetic_piece  # noqa: E402
from timing_features import (  # noqa: E402
    assert_no_timing_leakage,
    featurize_for_humanizing,
    featurize_performance,
    smooth_beat_times,
    snap_beats,
)
from timing_sampling import (  # noqa: E402
    Z_MAX,
    CopulaParams,
    PortableRng,
    fit_copula,
    fit_sampler,
    normal_knots,
    normal_scores,
    quantile_function,
    sample_latent,
    sample_offsets,
)

MS_PER_TICK = 500.0 / TPB  # synthetic pieces run at 120 BPM


def _score() -> mido.MidiFile:
    """4/4 at 120 BPM, plus a trailing event so jittered files keep the same end tick."""
    mf = synthetic_piece(4, 8, [120])
    mf.tracks[1].append(mido.Message("control_change", control=1, value=0, time=2 * TPB))
    return mf


def _perform(score_md, offsets_ms: np.ndarray):
    """The score with every note moved by ``offsets_ms`` (durations kept), as a new MidiData."""
    shift = np.rint(offsets_ms / MS_PER_TICK)
    on = score_md.notes.onset_tick + shift
    off = score_md.notes.offset_tick + shift
    return parse_midi(with_timing(score_md, on, off))


def _offsets(score_md, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ms = np.clip(rng.normal(0, 12, len(score_md.notes)), -30, 30)
    return np.where(score_md.notes.onset_tick == 0, np.abs(ms), ms)  # nothing can start before tick 0


def test_smoothing_is_exact_for_constant_tempo_and_monotone() -> None:
    t = 0.5 * np.arange(40.0)
    assert np.allclose(smooth_beat_times(t, 3), t, atol=1e-12)
    rng = np.random.default_rng(1)
    wobbly = np.cumsum(rng.uniform(0.3, 0.7, 200))
    s = smooth_beat_times(wobbly, 2)
    assert np.all(np.diff(s) > 0)
    assert np.abs(s - wobbly).mean() < np.abs(wobbly - np.linspace(wobbly[0], wobbly[-1], 200)).mean()


def test_snap_uses_one_subdivision_per_beat() -> None:
    b = np.array([0.02, 0.27, 0.49, 0.77, 1.0, 1.34, 1.66, 2.0, 2.6, 3.31])
    q = snap_beats(b, [4, 3], 0.01)
    assert np.allclose(q, [0, 0.25, 0.5, 0.75, 1, 4 / 3, 5 / 3, 2, 2.5, 3.25])
    assert np.all(np.diff(q) >= 0)


def test_targets_recover_injected_offsets() -> None:
    score = parse_midi(_score())
    perf = _perform(score, _offsets(score))
    df = featurize_performance(perf, Config(), beat_source="midi").df
    # Pair each performed note with its score note (same pitch, nearest onset).
    p = df["note_order"].to_numpy()
    perf_tick = perf.notes.onset_tick[p]
    pitch = perf.notes.pitch[p]
    expected_tick = np.empty(len(p))
    for i, (pt, pi) in enumerate(zip(perf_tick, pitch)):
        cand = score.notes.onset_tick[score.notes.pitch == pi]
        expected_tick[i] = cand[np.argmin(np.abs(cand - pt))]
    assert np.allclose(df["score_beat"], expected_tick / TPB, atol=1e-9), "score positions not recovered"
    assert np.allclose(df["offset_ms"], (perf_tick - expected_tick) * MS_PER_TICK, atol=0.01)


def test_features_do_not_see_micro_timing() -> None:
    cfg = Config()
    score = parse_midi(_score())
    a = featurize_performance(score, cfg, beat_source="midi")
    for seed in (1, 2):
        b = featurize_performance(_perform(score, _offsets(score, seed)), cfg, beat_source="midi")
        cols = list(a.groups)
        assert a.df[cols].equals(b.df[cols]), f"features changed with micro-timing (seed {seed})"
        assert not np.allclose(b.df["offset_ms"], 0)


def test_quantized_input_uses_its_own_grid() -> None:
    cfg = Config()
    score = parse_midi(_score())
    piece = featurize_for_humanizing(score, cfg)
    assert piece.score is None and piece.grid.source == "midi"
    perf = parse_midi(synthetic_piece(4, 8, [110], jitter_s=0.015))
    assert featurize_for_humanizing(perf, cfg).score is not None


def test_retiming_roundtrip_keeps_other_events() -> None:
    md = parse_midi(_score())
    shift = np.where(np.arange(len(md.notes)) % 3 == 0, 30, -12)
    on = md.notes.onset_tick + np.where(md.notes.onset_tick == 0, 0, shift)
    off = md.notes.offset_tick + np.where(md.notes.onset_tick == 0, 0, shift)
    out = parse_midi(with_timing(md, on, off))
    key = lambda n: sorted(zip(n.onset_tick.tolist(), n.pitch.tolist(), n.velocity.tolist()))  # noqa: E731
    expected = sorted(zip(on.astype(int).tolist(), md.notes.pitch.tolist(), md.notes.velocity.tolist()))
    assert key(out.notes) == expected
    cc = lambda mf: [(t, m.control) for t, m in _abs(mf) if m.type == "control_change"]  # noqa: E731
    assert cc(out.midi) == cc(md.midi)
    warped = with_timing(md, on, off, warp=lambda t: t + 100)
    assert [t for t, _ in cc(warped)] == [t + 100 for t, _ in cc(md.midi)]


def _abs(mf: mido.MidiFile) -> list:
    out = []
    for tr in mf.tracks:
        t = 0
        for m in tr:
            t += m.time
            out.append((t, m))
    return sorted(out, key=lambda e: e[0])


def test_same_key_notes_never_cut_each_other() -> None:
    mf = mido.MidiFile(ticks_per_beat=TPB)
    tr = mido.MidiTrack()
    for t in (0, TPB):
        tr.append(mido.Message("note_on", note=60, velocity=80, time=0 if t == 0 else TPB - 400))
        tr.append(mido.Message("note_on", note=60, velocity=0, time=400))
    mf.tracks.append(tr)
    md = parse_midi(mf)
    # Push the first note late and the second early: they would overlap.
    out = parse_midi(with_timing(md, md.notes.onset_tick + [200, -150], md.notes.offset_tick + [200, -150]))
    assert len(out.notes) == 2
    assert out.notes.offset_tick[0] <= out.notes.onset_tick[1]
    assert out.notes.offset_tick[1] - out.notes.onset_tick[1] == 400  # second note intact


def test_quantile_function_roundtrip() -> None:
    alphas = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    knots = normal_knots(alphas)
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(0, 20, (500, len(alphas))), axis=1) + np.linspace(-30, 30, len(alphas))
    z = rng.uniform(-Z_MAX, Z_MAX, 500)
    y = quantile_function(q, knots, z)
    assert np.allclose(normal_scores(q, knots, y), z, atol=1e-9)
    assert np.allclose(quantile_function(q, knots, np.zeros(500)), q[:, 3])
    assert np.allclose(quantile_function(q, knots, np.full(500, knots[-1])), q[:, -1])


def test_portable_rng_matches_splitmix64() -> None:
    u = PortableRng(0).uniforms(2)
    # SplitMix64 reference outputs for seed 0: 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4
    assert u[0] == ((0xE220A8397B1DCDAF >> 11) + 0.5) / 2.0**53
    assert u[1] == ((0x6E789E6AA1B965F4 >> 11) + 0.5) / 2.0**53
    z = PortableRng(5).normals(200001)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    a, b = PortableRng(9), PortableRng(9)
    a.normals(3)
    a.normals(4)
    b.uniforms(4)  # 3 normals use 4 uniforms (an odd count drops the last sine)
    b.normals(4)
    assert np.array_equal(a.normals(5), b.normals(5))


def test_copula_marginals_correlations_and_fit() -> None:
    rng = PortableRng(0)
    groups = np.repeat(np.arange(20000), 3)  # 3-note chords every half beat
    beats = groups * 0.5
    true = CopulaParams(rho=0.7, ell=1.5)
    z = sample_latent(groups, beats, true, rng)
    assert abs(z.std() - 1) < 0.02
    zz = z.reshape(-1, 3)
    assert abs(np.corrcoef(zz[:, 0], zz[:, 1])[0, 1] - 0.7) < 0.03
    lag2 = np.corrcoef(zz[:-2, 0], zz[2:, 0])[0, 1]  # one beat apart
    assert abs(lag2 - 0.7 * np.exp(-1 / 1.5)) < 0.03
    fit = fit_copula(z, np.zeros(len(z), dtype=int), groups, beats)
    assert abs(fit.rho - 0.7) < 0.03 and abs(fit.ell - 1.5) / 1.5 < 0.15


def test_recalibration_widens_a_too_narrow_model() -> None:
    gen = np.random.default_rng(0)
    rng = PortableRng(1)
    alphas = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    knots = normal_knots(alphas)
    n = 30000
    q = np.tile(10.0 * knots, (n, 1))  # predicts N(0, 10 ms) ...
    y = gen.normal(0, 20.0, n)  # ... but offsets are N(0, 20 ms)
    groups = np.arange(n)
    params = fit_sampler(normal_scores(q, knots, y), np.zeros(n, dtype=int), groups, groups * 0.5)
    draws = sample_offsets(q, alphas, groups, groups * 0.5, params, 1.0, rng)
    assert abs(draws.std() / 20.0 - 1) < 0.1, draws.std()
    raw = sample_offsets(q, alphas, groups, groups * 0.5, CopulaParams(rho=params.rho, ell=params.ell), 1.0, rng)
    assert raw.std() < 12


def test_timing_leakage_guard() -> None:
    assert_no_timing_leakage(["pitch", "ioi_prev_beats", "chord_size"])
    for bad in (["offset_ms"], ["perf_onset_s"], ["score_beat"], ["pitch", "velocity"], ["ref_time_s"]):
        try:
            assert_no_timing_leakage(bad)
        except ValueError:
            continue
        raise AssertionError(f"timing leakage guard accepted {bad}")


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
