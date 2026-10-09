"""Timing pipeline: score reconstruction, micro-timing targets, and features.

How a MAESTRO performance becomes training data
-----------------------------------------------
1. Raw grid: beats tracked from the performance (``beat_tracking``). It follows rubato.
2. Score position: every onset group (chord: onsets within ``features.chord_tolerance_s``
   of its first note) is snapped *as a whole* to the raw grid. Chord-mates share one
   score position, so chord asynchrony (melody lead, rolled chords) ends up in the target.
   Each beat is quantized to ONE subdivision, 1/4 or 1/3 (``timing.quantize_subdivisions``),
   chosen from the chords in that beat. Snapping every note to the union of both grids
   would give lopsided snapping cells around the 1/4, 1/3, 2/3 and 3/4 positions, which
   shows up as fake systematic timing (measured on MAESTRO: notes on the 2nd and 4th 16th
   would average -20 and +20 ms).
3. Reference beat: the raw beat times smoothed by a local linear fit over
   +/- ``timing.reference_smoothing_beats``. The target is

       offset_ms = 1000 * (performed onset - reference_beat_time(score position))

   Tempo changes slower than that window count as tempo, not timing, and are not
   predicted (in a DAW the host tempo map owns them).
4. Score timeline: the raw beats smoothed over +/- ``timing.score_smoothing_beats``.
   Features are extracted from the score positions mapped through this timeline. It
   keeps section-level tempo, like a DAW tempo map, but none of the beat-to-beat timing
   the target is made of. Performed onset times never reach the features.

At inference on quantized DAW/score MIDI, the file's own tempo map is the score timeline,
and the predicted offsets are added to the notes' times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from beat_tracking import BeatGrid, build_beat_grid
from config import Config
from midi_features import _group_onsets, assert_no_leakage, extract_features
from midi_io import MidiData, looks_quantized

log = logging.getLogger(__name__)

TIMING_TARGET_COLUMNS = ["offset_ms"]
# Per-note bookkeeping stored next to the features; never model inputs.
TIMING_META_COLUMNS = ["score_beat", "perf_onset_s", "ref_time_s"]


def assert_no_timing_leakage(columns: list[str]) -> None:
    """Refuse targets, performed times and bookkeeping columns as model inputs."""
    assert_no_leakage(columns)
    bad = [c for c in columns if c in TIMING_TARGET_COLUMNS or c in TIMING_META_COLUMNS or c.startswith("perf_")]
    if bad:
        raise ValueError(f"leakage guard: these columns may not be timing-model inputs: {bad}")


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def smooth_beat_times(times: np.ndarray, half: int) -> np.ndarray:
    """Local linear fit of beat time vs beat index over [i - half, i + half] (truncated at the ends).

    In the interior this is the moving average of the beat times, so it is exact for a
    constant tempo and strictly increasing whenever the input is.
    """
    t = np.asarray(times, dtype=np.float64)
    n = len(t)
    if half <= 0 or n < 3:
        return t.copy()
    x = np.arange(n, dtype=np.float64)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)

    def wsum(a: np.ndarray) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(a)])
        return c[hi] - c[lo]

    m, sx, sy, sxx, sxy = wsum(np.ones(n)), wsum(x), wsum(t), wsum(x * x), wsum(x * t)
    slope = (m * sxy - sx * sy) / (m * sxx - sx * sx)
    out = (sy - slope * sx) / m + slope * x
    if np.any(np.diff(out) <= 0):  # only possible at the truncated ends of a pathological grid
        log.debug("smoothed beat grid not increasing; keeping the raw beats")
        return t.copy()
    return out


def smoothed_grid(grid: BeatGrid, half: int) -> BeatGrid:
    """Same beats and meter, smoothed beat times."""
    return BeatGrid(smooth_beat_times(grid.times, half), grid.beat_in_bar, grid.beats_per_bar, grid.beat_unit,
                    grid.source, grid.meter_estimated)


def snap_beats(beats: np.ndarray, subdivisions: list[int], switch_cost: float = 0.0) -> np.ndarray:
    """Snap positions (in beats) to a grid chosen per beat.

    All positions within one beat (same integer part) use one subdivision: the first in
    ``subdivisions`` unless another lowers that beat's total squared snapping error
    (beats^2) by more than ``switch_cost``. Within a beat the snapping cells are then
    symmetric, and the result is monotone, so order is preserved.
    """
    b = np.asarray(beats, dtype=np.float64)
    cands = [np.rint(b * s) / s for s in subdivisions]
    if len(subdivisions) == 1 or len(b) == 0:
        return cands[0]
    _, inv = np.unique(np.floor(b), return_inverse=True)
    choice = np.zeros(inv.max() + 1, dtype=np.int64)
    best = np.bincount(inv, weights=(cands[0] - b) ** 2)
    for j in range(1, len(subdivisions)):
        err = np.bincount(inv, weights=(cands[j] - b) ** 2, minlength=len(best)) + switch_cost
        better = err < best
        choice[better], best[better] = j, err[better]
    return np.choose(choice[inv], cands)


def snap_groups(onset: np.ndarray, grid: BeatGrid, subdivisions: list[int], switch_cost: float,
                chord_tol: float) -> np.ndarray:
    """Score position (beats) of every note: each onset group is snapped as a whole, at its mean onset."""
    onset = np.asarray(onset, dtype=np.float64)
    by_time = np.argsort(onset, kind="stable")
    gid = np.empty(len(onset), dtype=np.int64)
    gid[by_time] = _group_onsets(onset[by_time], chord_tol)
    mean_on = np.bincount(gid, weights=onset) / np.bincount(gid)
    return snap_beats(grid.time_to_beat(mean_on), subdivisions, switch_cost)[gid]


# ---------------------------------------------------------------------------
# Performance -> (score, offsets)
# ---------------------------------------------------------------------------


@dataclass
class TimingScore:
    """A performance split into a score and its timing. Arrays are in ``md.notes`` order."""

    onset_beat: np.ndarray  # score position of each onset (beats of the raw grid)
    offset_beat: np.ndarray  # quantized release position
    raw_grid: BeatGrid  # tracked beats, follows the performance
    reference_grid: BeatGrid  # the local beat offsets are measured from
    score_grid: BeatGrid  # smooth timeline the features (and renders) use

    @property
    def score_onset(self) -> np.ndarray:
        return self.score_grid.beat_to_time(self.onset_beat)

    @property
    def score_offset(self) -> np.ndarray:
        return self.score_grid.beat_to_time(self.offset_beat)

    def warp_seconds(self, t: np.ndarray) -> np.ndarray:
        """Map performance time to score time (for events that are not notes, e.g. pedals)."""
        return self.score_grid.beat_to_time(self.raw_grid.time_to_beat(t))


def reconstruct_score(md: MidiData, cfg: Config, beat_source: str | None = None) -> TimingScore:
    """Tracked beats -> chord-wise quantized score positions + the reference and score grids."""
    beat_cfg = cfg.beat if beat_source is None else replace(cfg.beat, source=beat_source)
    raw = build_beat_grid(md, beat_cfg)
    tc = cfg.timing
    subs = list(tc.quantize_subdivisions)
    on_b = snap_groups(md.notes.onset, raw, subs, tc.subdivision_switch_cost, cfg.features.chord_tolerance_s)
    off_b = snap_beats(raw.time_to_beat(md.notes.offset), subs, tc.subdivision_switch_cost)
    off_b = np.maximum(off_b, on_b + 1.0 / max(subs))
    return TimingScore(on_b, off_b, raw, smoothed_grid(raw, tc.reference_smoothing_beats),
                       smoothed_grid(raw, tc.score_smoothing_beats))


@dataclass
class TimingPiece:
    df: pd.DataFrame  # canonical order: meta + features (+ targets for performances)
    groups: dict[str, str]  # feature -> group
    grid: BeatGrid  # the timeline the features (and output times) live on
    score: TimingScore | None  # set when the input was a performance


def featurize_performance(md: MidiData, cfg: Config, beat_source: str | None = None) -> TimingPiece:
    """Training/evaluation path: features from the reconstructed score, targets from the performance."""
    sc = reconstruct_score(md, cfg, beat_source)
    df, groups = extract_features(sc.score_onset, sc.score_offset, md.notes.pitch, sc.score_grid, cfg.features)
    order = df["note_order"].to_numpy()
    score_beat = sc.onset_beat[order]
    perf = md.notes.onset[order]
    ref = sc.reference_grid.beat_to_time(score_beat)
    extra = pd.DataFrame({
        "score_beat": score_beat,
        "perf_onset_s": perf,
        "ref_time_s": ref,
        "offset_ms": ((perf - ref) * 1000.0).astype(np.float32),
    })
    return TimingPiece(pd.concat([df, extra], axis=1), groups, sc.score_grid, sc)


def featurize_quantized(md: MidiData, cfg: Config, beat_source: str = "midi", snap: bool = False) -> TimingPiece:
    """Inference path for DAW/score MIDI: the file's notes are the score.

    ``snap`` first moves each onset group to the nearest grid subdivision (for files that
    are almost, but not exactly, quantized). The grid is the file's own tempo/time-signature
    map with ``beat_source="midi"``.
    """
    grid = build_beat_grid(md, replace(cfg.beat, source=beat_source))
    onset, offset = md.notes.onset, md.notes.offset
    if snap:
        subs, cost = list(cfg.timing.quantize_subdivisions), cfg.timing.subdivision_switch_cost
        on_b = snap_groups(onset, grid, subs, cost, cfg.features.chord_tolerance_s)
        off_b = np.maximum(snap_beats(grid.time_to_beat(offset), subs, cost), on_b + 1.0 / max(subs))
        onset, offset = grid.beat_to_time(on_b), grid.beat_to_time(off_b)
    df, groups = extract_features(onset, offset, md.notes.pitch, grid, cfg.features)
    return TimingPiece(df, groups, grid, None)


def featurize_for_humanizing(md: MidiData, cfg: Config, beat_source: str = "auto", snap: bool = False) -> TimingPiece:
    """Quantized input -> its own grid; a played performance -> reconstructed score (its timing is replaced)."""
    quantized = looks_quantized(md, cfg.beat.auto_quantized_fraction)
    if beat_source == "midi" or (beat_source == "auto" and quantized):
        return featurize_quantized(md, cfg, "midi", snap=snap)
    return featurize_performance(md, cfg, "tracked")
