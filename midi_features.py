"""Per-note feature extraction, targets, and the shared MIDI -> features pipeline.

Leakage rules enforced here
---------------------------
* ``extract_features`` does not accept velocities at all: every input feature
  is a function of onset/offset times, pitches and the beat grid only.
* Targets (velocity, residual baseline, delta) are computed separately by
  ``compute_targets``. The residual baseline uses *other* notes' velocities
  (own chord excluded); it is part of the target transform, never an input.
* ``performance_conditioned_features`` (neighbouring/chord-mate velocities)
  are written with a ``pc_`` prefix and are rejected by ``assert_no_leakage``
  unless the run is explicitly the ``performance_conditioned`` experiment.

Canonical note order
--------------------
Notes are grouped into onset groups (chords: onsets within
``chord_tolerance_s`` of the group's first onset) and sorted by
(group, pitch). All windowed/sequential features use this order, which is
also the order rows are stored in. ``note_order`` maps each row back to its
index in ``midi_io.NoteArray`` so predictions can be written to the file.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from beat_tracking import BeatGrid, build_beat_grid
from config import Config, FeatureConfig, TargetConfig
from midi_io import MidiData

FEATURE_GROUP_ORDER = ["core", "note", "melodic", "local", "poly", "structure", "piece"]

# Ablation feature sets (cumulative).
FEATURE_SETS: dict[str, list[str]] = {
    "A": ["core"],
    "B": ["core", "note", "melodic", "local"],
    "C": ["core", "note", "melodic", "local", "poly"],
    "D": FEATURE_GROUP_ORDER,
}
FEATURE_SET_DESCRIPTIONS = {
    "A": "pitch + beat position",
    "B": "note + melodic/local context",
    "C": "note + local + chord/polyphony",
    "D": "full (adds structure + piece-level)",
}
FEATURE_SETS["full"] = FEATURE_SETS["D"]

META_COLUMNS = ["note_idx", "note_order", "group_id", "onset_s", "group_time_s"]
TARGET_COLUMNS = ["velocity", "baseline_velocity", "velocity_delta"]
PC_PREFIX = "pc_"
PC_COLUMNS = [
    "pc_prev_group_mean_velocity",
    "pc_next_group_mean_velocity",
    "pc_chordmates_mean_velocity",
    "pc_local_baseline_velocity",
]


def select_feature_columns(feature_groups: dict[str, str], feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature set '{feature_set}' (valid: {sorted(FEATURE_SETS)})")
    wanted = set(FEATURE_SETS[feature_set])
    return [c for c, g in feature_groups.items() if g in wanted]


def assert_no_leakage(columns: list[str], experiment: str = "primary") -> None:
    """Refuse any input column derived from performed velocity in the primary experiment."""
    if experiment == "performance_conditioned":
        bad = [c for c in columns if c in TARGET_COLUMNS]
    else:
        bad = [c for c in columns if "velocity" in c or c.startswith(PC_PREFIX) or c in TARGET_COLUMNS]
    bad += [c for c in columns if c in META_COLUMNS]
    if bad:
        raise ValueError(f"leakage guard: these columns may not be model inputs: {bad}")


# ---------------------------------------------------------------------------
# Small vectorised helpers
# ---------------------------------------------------------------------------


class _RangeMinMax:
    """Sparse table for O(1) range min/max queries over half-open index ranges."""

    def __init__(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.max_t = [x]
        self.min_t = [x]
        j = 1
        while (1 << j) <= len(x):
            half = 1 << (j - 1)
            pm, pn = self.max_t[-1], self.min_t[-1]
            m = np.maximum(pm[:-half], pm[half:])
            n = np.minimum(pn[:-half], pn[half:])
            pad = len(x) - len(m)
            self.max_t.append(np.concatenate([m, np.full(pad, -np.inf)]))
            self.min_t.append(np.concatenate([n, np.full(pad, np.inf)]))  # padding is never queried
            j += 1
        self.max_t = np.vstack(self.max_t)
        self.min_t = np.vstack(self.min_t)

    def query(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        length = np.maximum(hi - lo, 1)
        k = np.floor(np.log2(length)).astype(np.int64)
        right = hi - (1 << k)
        mx = np.maximum(self.max_t[k, lo], self.max_t[k, right])
        mn = np.minimum(self.min_t[k, lo], self.min_t[k, right])
        return mx, mn


def _window_view(x: np.ndarray, before: int, after: int, fill: float = np.nan) -> np.ndarray:
    """(n, before+after+1) view of x[i-before : i+after+1] with ``fill`` outside the array."""
    padded = np.concatenate([np.full(before, fill), np.asarray(x, dtype=np.float64), np.full(after, fill)])
    return np.lib.stride_tricks.sliding_window_view(padded, before + after + 1)


def _cumsum0(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])


def _take(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def _group_onsets(sorted_onsets: np.ndarray, tol: float) -> np.ndarray:
    """Chord grouping anchored on each group's first onset (so fast arpeggios do not chain)."""
    ids = np.empty(len(sorted_onsets), dtype=np.int64)
    cur = 0
    start = sorted_onsets[0]
    for i, t in enumerate(sorted_onsets):
        if t - start > tol:
            cur += 1
            start = t
        ids[i] = cur
    return ids


def _nearest_pitch_in_group(key: np.ndarray, g: np.ndarray, p: np.ndarray, target_g: np.ndarray, n_groups: int) -> np.ndarray:
    """Index of the note in group ``target_g`` whose pitch is nearest to ``p`` (-1 if none).

    Relies on ``key = group * 128 + pitch`` being sorted (canonical order).
    """
    n = len(key)
    q = target_g * 128 + p
    j = np.searchsorted(key, q)
    hi = np.clip(j, 0, n - 1)
    lo = np.clip(j - 1, 0, n - 1)
    ok_hi = g[hi] == target_g
    ok_lo = g[lo] == target_g
    d_hi = np.where(ok_hi, np.abs(p[hi] - p), np.inf)
    d_lo = np.where(ok_lo, np.abs(p[lo] - p), np.inf)
    res = np.where(d_lo <= d_hi, lo, hi)
    valid = (target_g >= 0) & (target_g < n_groups) & (ok_hi | ok_lo)
    return np.where(valid, res, -1)


def _near_grid(frac: np.ndarray, subdivision: int, tol: float) -> np.ndarray:
    x = frac * subdivision
    return np.abs(x - np.rint(x)) / subdivision <= tol


class _Columns:
    def __init__(self) -> None:
        self.data: dict[str, np.ndarray] = {}
        self.groups: dict[str, str] = {}

    def add(self, name: str, values: np.ndarray, group: str) -> None:
        if name in self.data:
            raise KeyError(f"duplicate feature {name}")
        self.data[name] = np.asarray(values, dtype=np.float32)
        self.groups[name] = group


def _fmt(x: float) -> str:
    return f"{x:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    onset: np.ndarray,
    offset: np.ndarray,
    pitch: np.ndarray,
    grid: BeatGrid,
    cfg: FeatureConfig,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Score-like features for every note. No velocity argument, by design.

    Returns a DataFrame in canonical order (meta columns + float32 features)
    and a {feature: group} mapping.
    """
    causal = cfg.context == "causal"
    if cfg.context not in ("causal", "bidirectional"):
        raise ValueError(f"unknown context mode '{cfg.context}'")
    onset = np.asarray(onset, dtype=np.float64)
    offset = np.maximum(np.asarray(offset, dtype=np.float64), onset)
    pitch = np.asarray(pitch, dtype=np.int64)
    n = len(onset)
    if n == 0:
        raise ValueError("no notes")

    # --- onset groups and canonical order ---------------------------------
    by_time = np.lexsort((pitch, onset))
    gid = np.empty(n, dtype=np.int64)
    gid[by_time] = _group_onsets(onset[by_time], cfg.chord_tolerance_s)
    order = np.lexsort((pitch, gid))
    on, off, p, g = onset[order], offset[order], pitch[order], gid[order]
    dur = off - on
    n_groups = int(g[-1]) + 1
    g_start = np.searchsorted(g, np.arange(n_groups), side="left")
    g_end = np.searchsorted(g, np.arange(n_groups), side="right")
    g_size = g_end - g_start
    g_time = np.minimum.reduceat(on, g_start)
    g_top = p[g_end - 1].astype(np.float64)  # pitch-sorted within a group
    g_low = p[g_start].astype(np.float64)
    gt = g_time[g]  # chord-mates share one onset time for context features
    idx = np.arange(n)

    g_beat = grid.time_to_beat(g_time)
    gt_b = g_beat[g]
    on_b = grid.time_to_beat(on)
    off_b = grid.time_to_beat(off)
    dur_b = np.maximum(off_b - on_b, 0.0)

    cols = _Columns()

    # --- core: pitch + metrical position ----------------------------------
    nb = len(grid.times)
    bi = np.clip(np.floor(gt_b).astype(np.int64), 0, nb - 1)
    frac = gt_b - np.floor(gt_b)
    rb = np.clip(np.rint(gt_b).astype(np.int64), 0, nb - 1)
    tol = cfg.strong_beat_tolerance
    on_beat = np.abs(gt_b - np.rint(gt_b)) <= tol
    bib_r, bpb_r = grid.beat_in_bar[rb], grid.beats_per_bar[rb]
    is_downbeat = on_beat & (bib_r == 0)
    is_half_bar = on_beat & (bpb_r >= 4) & (bpb_r % 2 == 0) & (bib_r == bpb_r // 2)
    level = np.full(n, 5.0)  # 0 downbeat, 1 half-bar, 2 beat, 3 eighth, 4 16th/triplet, 5 other
    level[_near_grid(frac, 3, tol / 2) | _near_grid(frac, 4, tol / 2)] = 4
    level[_near_grid(frac, 2, tol)] = 3
    level[on_beat] = 2
    level[is_half_bar] = 1
    level[is_downbeat] = 0
    bpb = grid.beats_per_bar[bi].astype(np.float64)
    beat_in_measure = grid.beat_in_bar[bi] + frac

    cols.add("pitch", p, "core")
    cols.add("pitch_class", p % 12, "core")
    cols.add("octave", p // 12 - 1, "core")
    cols.add("beat_in_measure", beat_in_measure, "core")
    cols.add("beat_in_measure_norm", beat_in_measure / bpb, "core")
    cols.add("beat_frac", frac, "core")
    cols.add("metrical_level", level, "core")
    cols.add("on_beat", on_beat, "core")
    cols.add("strong_beat", is_downbeat | is_half_bar, "core")
    cols.add("is_downbeat", is_downbeat, "core")

    # --- note-level --------------------------------------------------------
    cols.add("dur_s", dur, "note")
    cols.add("dur_beats", dur_b, "note")
    cols.add("onset_beat", gt_b, "note")
    cols.add("beat_index", np.floor(gt_b), "note")
    cols.add("local_tempo_bpm", grid.tempo_bpm[bi], "note")
    cols.add("ts_num", bpb, "note")
    cols.add("ts_den", grid.beat_unit[bi], "note")

    # --- melodic / sequential context -------------------------------------
    key = g * 128 + p
    prev_i = _nearest_pitch_in_group(key, g, p, g - 1, n_groups)  # voice-leading predecessor
    next_i = _nearest_pitch_in_group(key, g, p, g + 1, n_groups) if not causal else np.full(n, -1)
    prev_interval = p - _take(p.astype(float), prev_i)
    next_interval = _take(p.astype(float), next_i) - p
    g_prev_t = np.concatenate([[np.nan], g_time[:-1]])[g]
    g_next_t = np.concatenate([g_time[1:], [np.nan]])[g]
    g_prev_b = np.concatenate([[np.nan], g_beat[:-1]])[g]
    g_next_b = np.concatenate([g_beat[1:], [np.nan]])[g]
    prev_dur = _take(dur, prev_i)

    cols.add("prev_interval", prev_interval, "melodic")
    cols.add("abs_prev_interval", np.abs(prev_interval), "melodic")
    cols.add("ioi_prev_s", gt - g_prev_t, "melodic")
    cols.add("ioi_prev_beats", gt_b - g_prev_b, "melodic")
    cols.add("prev_dur_s", prev_dur, "melodic")
    cols.add("dur_ratio_prev", np.log2((dur + 0.01) / (prev_dur + 0.01)), "melodic")
    if not causal:
        cols.add("next_interval", next_interval, "melodic")
        cols.add("abs_next_interval", np.abs(next_interval), "melodic")
        cols.add("ioi_next_s", g_next_t - gt, "melodic")
        cols.add("ioi_next_beats", g_next_b - gt_b, "melodic")
        cols.add("next_dur_s", _take(dur, next_i), "melodic")
        direction = np.nan_to_num(np.sign(prev_interval)) + np.nan_to_num(np.sign(next_interval))
        cols.add("melodic_direction", direction, "melodic")
        cols.add("is_local_peak", (prev_interval > 0) & (next_interval < 0), "melodic")
        cols.add("is_local_trough", (prev_interval < 0) & (next_interval > 0), "melodic")
    else:
        cols.add("melodic_direction", np.nan_to_num(np.sign(prev_interval)), "melodic")

    # Contour of the top line (skyline) around this onset.
    tp = np.concatenate([[np.nan, np.nan], g_top, [np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        before = np.nanmean(np.stack([tp[g], tp[g + 1]]), axis=0)  # groups g-2, g-1
        after = np.nanmean(np.stack([tp[g + 3], tp[g + 4]]), axis=0)  # groups g+1, g+2
    cols.add("top_line_trend", (g_top[g] - before) if causal else (after - before), "melodic")

    # Repeated pitches (repeated notes are usually played differently).
    sp = np.lexsort((on, p))
    same = np.diff(p[sp]) == 0
    dt = np.diff(on[sp])
    prev_same = np.empty(n)
    prev_same[sp] = np.concatenate([[np.nan], np.where(same, dt, np.nan)])
    cols.add("same_pitch_prev_dt", prev_same, "melodic")
    if not causal:
        next_same = np.empty(n)
        next_same[sp] = np.concatenate([np.where(same, dt, np.nan), [np.nan]])
        cols.add("same_pitch_next_dt", next_same, "melodic")

    # --- chord / polyphony -------------------------------------------------
    sorted_on = np.sort(on)
    sorted_off = np.sort(off)
    held = np.clip(
        np.searchsorted(sorted_on, gt, side="left") - np.searchsorted(sorted_off, gt, side="right"), 0, None
    )
    chord_size = g_size[g].astype(np.float64)
    polyphony = held + chord_size
    span = g_top[g] - g_low[g]
    rank_top = g_end[g] - 1 - idx
    rank_bottom = idx - g_start[g]
    ctol = cfg.chord_tolerance_s
    cols.add("chord_size", chord_size, "poly")
    cols.add(
        "n_onsets_within_tol",
        np.searchsorted(sorted_on, on + ctol, side="right") - np.searchsorted(sorted_on, on - ctol, side="left"),
        "poly",
    )
    cols.add("held_notes_at_onset", held, "poly")
    cols.add("polyphony_at_onset", polyphony, "poly")
    cols.add("chord_highest", g_top[g], "poly")
    cols.add("chord_lowest", g_low[g], "poly")
    cols.add("chord_span", span, "poly")
    with np.errstate(invalid="ignore", divide="ignore"):
        cols.add("chord_pos_norm", np.where(span > 0, (p - g_low[g]) / span, np.nan), "poly")
    cols.add("dist_from_chord_top", g_top[g] - p, "poly")
    cols.add("dist_from_chord_bottom", p - g_low[g], "poly")
    cols.add("chord_rank_from_top", rank_top, "poly")
    cols.add("chord_rank_from_bottom", rank_bottom, "poly")
    cols.add("is_chord_top", rank_top == 0, "poly")
    cols.add("is_chord_bottom", rank_bottom == 0, "poly")

    # --- local context windows --------------------------------------------
    cs_p, cs_dur, cs_poly = _cumsum0(p), _cumsum0(dur), _cumsum0(polyphony)
    cs_first = _cumsum0(idx == g_start[g])
    rmq = _RangeMinMax(p)

    def window_stats(times: np.ndarray, w: float, tag: str) -> None:
        lo = np.searchsorted(times, times - w, side="left")
        hi = np.searchsorted(times, times if causal else times + w, side="right")
        cnt = (hi - lo).astype(np.float64)
        width = w if causal else 2 * w
        mean_p = (cs_p[hi] - cs_p[lo]) / cnt
        mx, mn = rmq.query(lo, hi)
        cols.add(f"{tag}_note_density", cnt / width, "local")
        cols.add(f"{tag}_onset_density", (cs_first[hi] - cs_first[lo]) / width, "local")
        cols.add(f"{tag}_mean_pitch", mean_p, "local")
        cols.add(f"{tag}_pitch_rel_mean", p - mean_p, "local")
        cols.add(f"{tag}_pitch_range", mx - mn, "local")
        cols.add(f"{tag}_mean_dur", (cs_dur[hi] - cs_dur[lo]) / cnt, "local")
        cols.add(f"{tag}_mean_polyphony", (cs_poly[hi] - cs_poly[lo]) / cnt, "poly")

    for w in cfg.time_windows_s:
        window_stats(gt, float(w), f"win{_fmt(w * 1000)}ms")
    for w in cfg.beat_windows:
        window_stats(gt_b, float(w), f"win{_fmt(w)}b")

    cols.add(
        "notes_in_prev_beat",
        np.searchsorted(gt_b, gt_b, side="left") - np.searchsorted(gt_b, gt_b - 1.0, side="left"),
        "local",
    )
    if not causal:
        cols.add(
            "notes_in_next_beat",
            np.searchsorted(gt_b, gt_b + 1.0, side="right") - np.searchsorted(gt_b, gt_b, side="right"),
            "local",
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for nn in cfg.nearest_n:
            after_n = 0 if causal else nn
            win = _window_view(p, nn, after_n)
            valid = (~np.isnan(win)).sum(axis=1)
            tw = _window_view(gt, nn, after_n)
            cols.add(f"n{nn}_pitch_rel_median", p - np.nanmedian(win, axis=1), "melodic")
            cols.add(f"n{nn}_pitch_rel_mean", p - np.nanmean(win, axis=1), "melodic")
            cols.add(f"n{nn}_pitch_range", np.nanmax(win, axis=1) - np.nanmin(win, axis=1), "local")
            cols.add(f"n{nn}_pitch_pct", (win < p[:, None]).sum(axis=1) / valid, "local")
            cols.add(
                f"n{nn}_time_per_note",
                np.where(valid > 1, (np.nanmax(tw, axis=1) - np.nanmin(tw, axis=1)) / np.maximum(valid - 1, 1), np.nan),
                "local",
            )

    # --- structure: bars, rests, phrase-like segments ----------------------
    cols.add("measure_number", grid.measure_index[bi], "structure")
    sound_end_g = np.maximum.accumulate(off)[g_end - 1]  # when the keyboard goes silent after group k
    silence = g_time[1:] - sound_end_g[:-1]
    rest_start = np.concatenate([[True], silence >= cfg.rest_min_s])
    ioi_s = np.diff(g_time)
    ioi_b = np.diff(g_beat)
    phrase_start = np.concatenate(
        [[True], (silence >= cfg.rest_min_s) | ((ioi_b >= cfg.phrase_gap_beats) & (ioi_s >= cfg.phrase_gap_min_s))]
    )

    def segments(starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = np.flatnonzero(starts)
        seg = np.cumsum(starts) - 1
        last = np.concatenate([first[1:] - 1, [n_groups - 1]])
        return first[seg], last[seg]  # per group: first and last group of its segment

    r_first, r_last = segments(rest_start)
    cols.add("time_since_rest_s", (g_time - g_time[r_first])[g], "structure")
    ph_first, ph_last = segments(phrase_start)
    since_b = (g_beat - g_beat[ph_first])[g]
    cols.add("time_since_phrase_start_beats", since_b, "structure")
    cols.add("notes_since_phrase_start", idx - g_start[ph_first[g]], "structure")
    if not causal:
        cols.add("time_to_rest_s", (sound_end_g[r_last] - g_time)[g], "structure")
        to_b = (g_beat[ph_last] - g_beat)[g]
        length = since_b + to_b
        with np.errstate(invalid="ignore", divide="ignore"):
            cols.add("phrase_pos_norm", np.where(length > 0, since_b / length, np.nan), "structure")
        cols.add("time_to_phrase_end_beats", to_b, "structure")
        cols.add("phrase_length_beats", length, "structure")
        cols.add("notes_to_phrase_end", g_end[ph_last[g]] - 1 - idx, "structure")
        total = max(float(g_time[-1] - g_time[0]), 1e-6)
        cols.add("rel_position", (gt - g_time[0]) / total, "structure")

    # --- piece-level (constant per file; whole-piece knowledge) -------------
    if cfg.include_piece_features and not causal:
        length_s = max(float(off.max() - on.min()), 1e-3)
        in_piece = (grid.times >= on.min()) & (grid.times <= off.max())
        tempo = grid.tempo_bpm[in_piece] if in_piece.any() else grid.tempo_bpm
        cols.add("piece_duration_s", np.full(n, length_s), "piece")
        cols.add("piece_note_rate", np.full(n, n / length_s), "piece")
        cols.add("piece_mean_pitch", np.full(n, p.mean()), "piece")
        cols.add("piece_pitch_std", np.full(n, p.std()), "piece")
        cols.add("piece_mean_polyphony", np.full(n, polyphony.mean()), "piece")
        cols.add("piece_mean_chord_size", np.full(n, g_size.mean()), "piece")
        cols.add("piece_median_tempo_bpm", np.full(n, np.median(tempo)), "piece")
        cols.add("pitch_rel_piece_mean", p - p.mean(), "piece")

    df = pd.DataFrame(
        {
            "note_idx": idx.astype(np.int32),
            "note_order": order.astype(np.int32),
            "group_id": g.astype(np.int32),
            "onset_s": on,
            "group_time_s": gt,
            **cols.data,
        }
    )
    return df, cols.groups


# ---------------------------------------------------------------------------
# Targets and the (optional, clearly separated) performance-conditioned inputs
# ---------------------------------------------------------------------------


def residual_baseline(velocity: np.ndarray, group_id: np.ndarray, window: int, stat: str = "median") -> np.ndarray:
    """Rolling statistic of surrounding velocities (canonical order), own chord excluded."""
    v = np.asarray(velocity, dtype=np.float64)
    win = _window_view(v, window, window)
    same = _window_view(group_id.astype(np.float64), window, window, fill=-1.0) == group_id[:, None]
    win = np.where(same, np.nan, win)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = np.nanmedian(win, axis=1) if stat == "median" else np.nanmean(win, axis=1)
    return np.where(np.isnan(base), np.median(v), base)


def compute_targets(velocity: np.ndarray, group_id: np.ndarray, cfg: TargetConfig) -> pd.DataFrame:
    """``velocity`` must be in canonical order (i.e. ``md.notes.velocity[df.note_order]``)."""
    v = np.asarray(velocity, dtype=np.float64)
    base = residual_baseline(v, np.asarray(group_id), cfg.residual_window_notes, cfg.residual_stat)
    return pd.DataFrame(
        {
            "velocity": v.astype(np.int16),
            "baseline_velocity": base.astype(np.float32),
            "velocity_delta": (v - base).astype(np.float32),
        }
    )


def performance_conditioned_features(velocity: np.ndarray, group_id: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    """Neighbour/chord-mate velocities. ONLY for the labelled 'performance_conditioned' experiment."""
    v = np.asarray(velocity, dtype=np.float64)
    g = np.asarray(group_id, dtype=np.int64)
    n_groups = int(g.max()) + 1
    gsum = np.bincount(g, weights=v, minlength=n_groups)
    gcnt = np.bincount(g, minlength=n_groups).astype(np.float64)
    gmean = gsum / gcnt
    with np.errstate(invalid="ignore", divide="ignore"):
        mates = np.where(gcnt[g] > 1, (gsum[g] - v) / (gcnt[g] - 1), np.nan)
    return pd.DataFrame(
        {
            "pc_prev_group_mean_velocity": np.concatenate([[np.nan], gmean[:-1]])[g],
            "pc_next_group_mean_velocity": np.concatenate([gmean[1:], [np.nan]])[g],
            "pc_chordmates_mean_velocity": mates,
            "pc_local_baseline_velocity": baseline,
        }
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Optional quantization (score-like second experiment)
# ---------------------------------------------------------------------------


def quantize_notes(
    onset: np.ndarray, offset: np.ndarray, grid: BeatGrid, subdivisions: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Snap onsets/offsets to the nearest grid subdivision (in beats), keeping the tempo curve."""

    def snap(b: np.ndarray) -> np.ndarray:
        best = np.full(len(b), np.nan)
        best_err = np.full(len(b), np.inf)
        for s in subdivisions:
            q = np.rint(b * s) / s
            err = np.abs(q - b)
            better = err < best_err
            best[better], best_err[better] = q[better], err[better]
        return best

    ob = snap(grid.time_to_beat(onset))
    eb = snap(grid.time_to_beat(offset))
    eb = np.maximum(eb, ob + 1.0 / max(subdivisions))
    return grid.beat_to_time(ob), grid.beat_to_time(eb)


# ---------------------------------------------------------------------------
# Shared MIDI -> features pipeline (used by build, evaluate and humanize)
# ---------------------------------------------------------------------------


@dataclass
class PieceFeatures:
    df: pd.DataFrame
    groups: dict[str, str]
    grid: BeatGrid


def featurize_midi(md: MidiData, cfg: Config, beat_source: str | None = None) -> PieceFeatures:
    """Features for a parsed file. Reads onsets/offsets/pitches only - never velocities."""
    beat_cfg = cfg.beat if beat_source is None else replace(cfg.beat, source=beat_source)
    grid = build_beat_grid(md, beat_cfg)
    onset, offset = md.notes.onset, md.notes.offset
    if cfg.features.quantize:
        onset, offset = quantize_notes(onset, offset, grid, list(cfg.features.quantize_subdivisions))
    df, groups = extract_features(onset, offset, md.notes.pitch, grid, cfg.features)
    return PieceFeatures(df, groups, grid)


def canonical_velocities(md: MidiData, df: pd.DataFrame) -> np.ndarray:
    return md.notes.velocity[df["note_order"].to_numpy()].astype(np.float64)


def to_note_order(values_canonical: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Re-index canonical-order values to ``md.notes`` order (for writing MIDI)."""
    out = np.empty(len(values_canonical), dtype=np.float64)
    out[df["note_order"].to_numpy()] = values_canonical
    return out
