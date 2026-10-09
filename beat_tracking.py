"""Beat grids: from a MIDI tempo map, or estimated from note onsets.

Why a tracker is needed: MAESTRO files are Disklavier recordings whose tempo
map is a placeholder (120 BPM, 4/4, a single event). Bar/beat positions read
from that map are just ``seconds * 2`` and carry no musical meaning, so for
MAESTRO the beat grid is estimated from the notes themselves.

The tracker only ever sees onset times, key-down durations and pitches. It
has no access to velocity, so tracked beat/bar features cannot leak the target.

Method (deliberately simple and dependency-free):
1. onset envelope at ``frame_rate`` Hz from note onsets (weighted by note count
   and duration, log-compressed);
2. global tempo from the envelope autocorrelation with a log-normal prior,
   then a windowed local tempo curve constrained around it;
3. Ellis (2007) dynamic-programming beat tracking with a *time-varying* target
   period, so the beat follows rubato;
4. beats snapped to nearby chord onsets;
5. meter (3 vs 4 beats per bar) and downbeat phase from per-beat salience
   (onset count, long notes, bass notes, harmonic change), with the phase
   re-estimated per segment by Viterbi so a skipped beat does not corrupt
   the rest of the piece.

Limitations (documented, expected): the tracked "beat" is the tactus the
tracker settles on, which can be half or double the notated beat; meter and
downbeat estimates are heuristic. Bar-level features from tracked grids are
therefore noisier than from a real score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config import BeatConfig
from midi_io import MAX_GRID_BEATS, MidiData, MidiParseError, looks_quantized

log = logging.getLogger(__name__)


@dataclass
class BeatGrid:
    times: np.ndarray  # beat times in seconds, strictly increasing, len >= 2
    beat_in_bar: np.ndarray  # 0 = downbeat
    beats_per_bar: np.ndarray  # time-signature numerator (or estimated meter)
    beat_unit: np.ndarray  # time-signature denominator (4 when estimated)
    source: str  # "midi" or "tracked"
    meter_estimated: bool
    measure_index: np.ndarray = field(init=False)  # 1-based bar number; 0 = pickup
    tempo_bpm: np.ndarray = field(init=False)  # smoothed local tempo per beat

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("beat times must be strictly increasing with at least two beats")
        self.beat_in_bar = np.asarray(self.beat_in_bar, dtype=np.int32)
        self.beats_per_bar = np.asarray(self.beats_per_bar, dtype=np.int32)
        self.beat_unit = np.asarray(self.beat_unit, dtype=np.int32)
        self.measure_index = np.cumsum(self.beat_in_bar == 0).astype(np.int32)
        ibi = np.diff(self.times)
        ibi = np.append(ibi, ibi[-1])
        self.tempo_bpm = 60.0 / _running_median(ibi, 2)

    def time_to_beat(self, t: np.ndarray | float) -> np.ndarray:
        """Fractional beat index for times in seconds (linear extrapolation outside the grid)."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        bt = self.times
        b = np.interp(t, bt, np.arange(len(bt), dtype=np.float64))
        lo, hi = t < bt[0], t > bt[-1]
        b[lo] = (t[lo] - bt[0]) / (bt[1] - bt[0])
        b[hi] = len(bt) - 1 + (t[hi] - bt[-1]) / (bt[-1] - bt[-2])
        return b

    def beat_to_time(self, b: np.ndarray | float) -> np.ndarray:
        b = np.atleast_1d(np.asarray(b, dtype=np.float64))
        bt = self.times
        n = len(bt)
        t = np.interp(b, np.arange(n, dtype=np.float64), bt)
        lo, hi = b < 0, b > n - 1
        t[lo] = bt[0] + b[lo] * (bt[1] - bt[0])
        t[hi] = bt[-1] + (b[hi] - (n - 1)) * (bt[-1] - bt[-2])
        return t

    def extended(self, t_start: float, t_end: float) -> BeatGrid:
        """Grid extrapolated (at the edge tempo) so every note time falls inside it."""
        times = list(self.times)
        bib = list(self.beat_in_bar)
        bpb = list(self.beats_per_bar)
        unit = list(self.beat_unit)
        ibi0 = max(float(np.median(np.diff(self.times[:5]))), 0.05)
        ibi1 = max(float(np.median(np.diff(self.times[-5:]))), 0.05)
        pre_t: list[float] = []
        pre_b: list[int] = []
        t, b = times[0], bib[0]
        while t > t_start and len(pre_t) < MAX_GRID_BEATS:
            t -= ibi0
            b = (b - 1) % bpb[0]
            pre_t.append(t)
            pre_b.append(b)
        k = len(pre_t)
        times = pre_t[::-1] + times
        bib = pre_b[::-1] + bib
        bpb = [bpb[0]] * k + bpb
        unit = [unit[0]] * k + unit
        while times[-1] < t_end and len(times) < MAX_GRID_BEATS:
            times.append(times[-1] + ibi1)
            bib.append((bib[-1] + 1) % bpb[-1])
            bpb.append(bpb[-1])
            unit.append(unit[-1])
        return BeatGrid(np.array(times), np.array(bib), np.array(bpb), np.array(unit), self.source, self.meter_estimated)

    @property
    def meter_label(self) -> str:
        num = int(np.bincount(self.beats_per_bar).argmax())
        den = int(np.bincount(self.beat_unit).argmax())
        return f"{num}/{den}{' (estimated)' if self.meter_estimated else ''}"


def _running_median(x: np.ndarray, half: int) -> np.ndarray:
    if len(x) <= 1 or half <= 0:
        return x.copy()
    pad = np.pad(x, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1), axis=1)


# ---------------------------------------------------------------------------
# Grid from the MIDI file's own tempo / time-signature map
# ---------------------------------------------------------------------------


def grid_from_midi(md: MidiData) -> BeatGrid:
    """One grid point per time-signature denominator unit, honouring every tempo and TS change.

    Each time-signature event is assumed to sit on a bar line (standard MIDI practice).
    """
    tpb = md.ticks_per_beat
    end_tick = int(md.end_tick)
    ts = md.time_signatures
    beat_ticks: list[float] = []
    bib: list[int] = []
    bpb: list[int] = []
    unit: list[int] = []
    for i, (tick0, num, den) in enumerate(ts):
        step = tpb * 4.0 / den
        seg_end = ts[i + 1][0] if i + 1 < len(ts) else end_tick + num * step
        k = 0
        x = float(tick0)
        while x < seg_end - 1e-9:
            beat_ticks.append(x)
            bib.append(k % num)
            bpb.append(num)
            unit.append(den)
            k += 1
            x = tick0 + k * step
            if len(beat_ticks) > MAX_GRID_BEATS:
                raise MidiParseError("tempo/time-signature grid is implausibly long")
    if len(beat_ticks) < 2:
        step = float(tpb)
        beat_ticks = [0.0, step]
        bib, bpb, unit = [0, 1], [4, 4], [4, 4]
    times = md.tempo_map.ticks_to_seconds(np.asarray(beat_ticks))
    return BeatGrid(times, np.array(bib), np.array(bpb), np.array(unit), "midi", meter_estimated=False)


# ---------------------------------------------------------------------------
# Onset-based beat tracking
# ---------------------------------------------------------------------------


def _onset_envelope(onset: np.ndarray, offset: np.ndarray, fr: float) -> np.ndarray:
    n_frames = int(np.ceil((onset.max() + 2.0) * fr)) + 1
    acc = np.zeros(n_frames)
    weight = 1.0 + np.minimum(offset - onset, 1.0)  # long notes are more beat-like
    np.add.at(acc, np.clip(np.rint(onset * fr).astype(np.int64), 0, n_frames - 1), weight)
    env = np.log1p(acc)  # big chords should not dominate everything
    kernel = np.exp(-0.5 * np.arange(-3, 4) ** 2)
    env = np.convolve(env, kernel / kernel.sum(), mode="same")
    return env / (env.std() + 1e-9)


def _autocorr_rows(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation of each row of x (rows are mean-removed)."""
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft, axis=-1)
    ac = np.fft.irfft(spec * np.conj(spec), nfft, axis=-1)[..., :n]
    return ac / (n - np.arange(n))


def _log_gauss(lags: np.ndarray, centre: float, octaves: float) -> np.ndarray:
    return np.exp(-0.5 * (np.log2(lags / centre) / octaves) ** 2)


def _global_period(env: np.ndarray, fr: float, cfg: BeatConfig) -> float:
    ac = _autocorr_rows(env[None, :])[0]
    lo = max(2, int(np.floor(fr * 60.0 / cfg.max_bpm)))
    hi = min(len(ac) - 2, int(np.ceil(fr * 60.0 / cfg.min_bpm)))
    if hi <= lo:
        return fr * 60.0 / cfg.tempo_prior_bpm
    lags = np.arange(lo, hi + 1)
    score = np.maximum(ac[lags], 0) * _log_gauss(lags.astype(float), fr * 60.0 / cfg.tempo_prior_bpm, cfg.tempo_prior_octaves)
    k = int(np.argmax(score))
    if score[k] <= 0:
        return fr * 60.0 / cfg.tempo_prior_bpm
    return float(lags[k] + _parabolic(score, k))


def _parabolic(y: np.ndarray, k: int) -> float:
    if 0 < k < len(y) - 1:
        a, b, c = y[k - 1], y[k], y[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 0.0


def _local_periods(env: np.ndarray, fr: float, global_period: float, cfg: BeatConfig) -> np.ndarray:
    n = len(env)
    win = max(int(cfg.local_tempo_window_s * fr), int(4 * global_period))
    hop = max(1, int(cfg.local_tempo_hop_s * fr))
    centres = np.arange(0, n, hop)
    padded = np.pad(env, (win // 2, win - win // 2))
    rows = np.lib.stride_tricks.sliding_window_view(padded, win)[centres]
    ac = _autocorr_rows(rows)
    r = cfg.local_tempo_max_ratio
    lo = max(2, int(np.floor(global_period / r)))
    hi = min(win - 2, int(np.ceil(global_period * r)))
    lags = np.arange(lo, hi + 1)
    score = np.maximum(ac[:, lags], 0) * _log_gauss(lags.astype(float), global_period, 0.3)[None, :]
    best = np.argmax(score, axis=1)
    local = lags[best].astype(float)
    weak = (score[np.arange(len(best)), best] <= 1e-6) | (rows.sum(axis=1) < 1e-6)
    local[weak] = np.nan
    if np.all(np.isnan(local)):
        return np.full(n, global_period)
    good = ~np.isnan(local)
    local = np.interp(centres, centres[good], local[good])
    local = _running_median(local, 2)
    return np.interp(np.arange(n), centres, local)


def _dp_beats(env: np.ndarray, period: np.ndarray, tightness: float, last_onset_frame: int) -> np.ndarray:
    """Ellis-style DP; predecessor window [t - 2p, t - p/2] with a log-period penalty."""
    n = len(env)
    pint = np.clip(np.rint(period), 2, None).astype(np.int64)
    cum = np.zeros(n)
    back = np.full(n, -1, dtype=np.int64)
    cost_cache: dict[int, np.ndarray] = {}
    for t in range(n):
        p = int(pint[t])
        hi = t - max(1, p // 2)
        if hi < 0:
            cum[t] = env[t]
            continue
        costs = cost_cache.get(p)
        if costs is None:
            taus = np.arange(2 * p, max(1, p // 2) - 1, -1, dtype=np.float64)
            costs = -tightness * np.log(taus / p) ** 2
            cost_cache[p] = costs
        lo = t - 2 * p
        c = costs
        if lo < 0:
            c = costs[-lo:]
            lo = 0
        cand = cum[lo : hi + 1] + c
        k = int(np.argmax(cand))
        best = cand[k]
        if best > 0:
            cum[t] = env[t] + best
            back[t] = lo + k
        else:
            cum[t] = env[t]
    p_end = int(pint[min(last_onset_frame, n - 1)])
    a = max(0, last_onset_frame - p_end)
    b = min(n, last_onset_frame + 2)
    t = a + int(np.argmax(cum[a:b]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = int(back[t])
    return np.array(beats[::-1], dtype=np.float64)


def _snap_to_onsets(beat_times: np.ndarray, onset_times: np.ndarray, frac: float) -> np.ndarray:
    if frac <= 0 or len(beat_times) < 2:
        return beat_times
    ibi = np.diff(beat_times)
    ibi = np.append(ibi, ibi[-1])
    j = np.searchsorted(onset_times, beat_times)
    lo = onset_times[np.clip(j - 1, 0, len(onset_times) - 1)]
    hi = onset_times[np.clip(j, 0, len(onset_times) - 1)]
    nearest = np.where(np.abs(beat_times - lo) <= np.abs(hi - beat_times), lo, hi)
    snapped = np.where(np.abs(nearest - beat_times) <= frac * ibi, nearest, beat_times)
    # Never let snapping reorder or merge beats.
    keep = np.concatenate([[True], np.diff(snapped) > 1e-3])
    snapped = np.where(keep, snapped, beat_times)
    if np.any(np.diff(snapped) <= 0):
        return beat_times
    return snapped


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def _beat_salience(beat_times: np.ndarray, onset: np.ndarray, offset: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """How downbeat-like each beat looks, from onsets/durations/pitches only."""
    nb = len(beat_times)
    ibi = np.diff(beat_times)
    ibi = np.append(ibi, ibi[-1])
    j = np.searchsorted(beat_times, onset)
    j0 = np.clip(j - 1, 0, nb - 1)
    j1 = np.clip(j, 0, nb - 1)
    nearest = np.where(np.abs(onset - beat_times[j0]) <= np.abs(onset - beat_times[j1]), j0, j1)
    near = np.abs(onset - beat_times[nearest]) <= 0.15 * ibi[nearest]
    idx = nearest[near]
    count = np.bincount(idx, minlength=nb).astype(float)
    dur_beats = (offset - onset) / ibi[nearest]
    long_notes = np.bincount(idx, weights=np.minimum(dur_beats[near], 4.0), minlength=nb)
    low = np.full(nb, 128.0)
    np.minimum.at(low, idx, pitch[near].astype(float))
    bass = np.where(count > 0, np.clip(np.median(pitch) - low, 0, 36) / 12.0, 0.0)
    # Harmonic change: pitch-class profile of each inter-beat span vs the previous span.
    starts = beat_times - 0.15 * ibi
    span = np.clip(np.searchsorted(starts, onset, side="right") - 1, 0, nb - 1)
    prof = np.zeros((nb, 12))
    np.add.at(prof, (span, pitch.astype(np.int64) % 12), np.minimum(offset - onset, 2.0))
    norm = np.linalg.norm(prof, axis=1, keepdims=True)
    prof = np.divide(prof, norm, out=np.zeros_like(prof), where=norm > 0)
    change = np.zeros(nb)
    both = (norm[1:, 0] > 0) & (norm[:-1, 0] > 0)
    change[1:] = np.where(both, 1.0 - np.sum(prof[1:] * prof[:-1], axis=1), 0.0)
    return _zscore(np.log1p(count)) + _zscore(np.log1p(long_notes)) + _zscore(bass) + _zscore(change)


def _estimate_meter(salience: np.ndarray, cfg: BeatConfig) -> tuple[int, np.ndarray]:
    """Return (beats_per_bar, beat_in_bar) using per-segment phase with a Viterbi path."""
    nb = len(salience)
    candidates = [cfg.fixed_meter] if cfg.fixed_meter else list(cfg.meter_candidates)
    best_fit, best_m, best_bib = -np.inf, candidates[0], np.arange(nb) % candidates[0]
    idx_all = np.arange(nb)
    for m in candidates:
        seg_len = max(m, (cfg.meter_segment_beats // m) * m)
        n_seg = int(np.ceil(nb / seg_len))
        contrast = np.zeros((n_seg, m))
        for k in range(n_seg):
            seg = salience[k * seg_len : (k + 1) * seg_len]
            idx = idx_all[k * seg_len : k * seg_len + len(seg)]
            sd = seg.std() + 1e-6
            for phi in range(m):
                mask = (idx - phi) % m == 0
                n_down = int(mask.sum())
                if n_down == 0 or n_down == len(seg):
                    continue
                se = sd * np.sqrt(1.0 / n_down - 1.0 / len(seg))
                contrast[k, phi] = (seg[mask].mean() - seg.mean()) / se
        penalty = cfg.meter_phase_change_penalty * (1.0 - np.eye(m))
        score = contrast[0].copy()
        back = np.zeros((n_seg, m), dtype=np.int64)
        for k in range(1, n_seg):
            trans = score[:, None] - penalty  # [from, to]
            back[k] = np.argmax(trans, axis=0)
            score = contrast[k] + trans[back[k], np.arange(m)]
        phase = np.zeros(n_seg, dtype=np.int64)
        phase[-1] = int(np.argmax(score))
        for k in range(n_seg - 1, 0, -1):
            phase[k - 1] = back[k, phase[k]]
        fit = float(score.max()) / n_seg
        if fit > best_fit:
            seg_phase = phase[np.minimum(idx_all // seg_len, n_seg - 1)]
            best_fit, best_m, best_bib = fit, m, (idx_all - seg_phase) % m
    return best_m, best_bib


def _regular_grid(start: float, end: float, bpm: float, meter: int) -> BeatGrid:
    ibi = 60.0 / bpm
    nb = max(2, int(np.ceil((end - start) / ibi)) + 1)
    times = start + ibi * np.arange(nb)
    return BeatGrid(times, np.arange(nb) % meter, np.full(nb, meter), np.full(nb, 4), "tracked", True)


def track_beats(onset: np.ndarray, offset: np.ndarray, pitch: np.ndarray, cfg: BeatConfig) -> BeatGrid:
    """Estimate a beat grid from note timing and pitch only (no velocity input by design)."""
    onset = np.asarray(onset, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    pitch = np.asarray(pitch)
    meter_default = cfg.fixed_meter or 4
    if len(onset) < 8 or onset.max() - onset.min() < 2.0:
        return _regular_grid(float(onset.min()), float(offset.max()) + 1.0, cfg.tempo_prior_bpm, meter_default)

    fr = cfg.frame_rate
    env = _onset_envelope(onset, offset, fr)
    gp = _global_period(env, fr, cfg)
    period = _local_periods(env, fr, gp, cfg)
    frames = _dp_beats(env, period, cfg.tightness, int(np.rint(onset.max() * fr)))
    if len(frames) < 4:
        return _regular_grid(float(onset.min()), float(offset.max()) + 1.0, 60.0 * fr / gp, meter_default)
    beat_times = _snap_to_onsets(frames / fr, np.unique(onset), cfg.snap_to_onset_frac)
    salience = _beat_salience(beat_times, onset, offset, pitch)
    meter, bib = _estimate_meter(salience, cfg)
    nb = len(beat_times)
    return BeatGrid(beat_times, bib, np.full(nb, meter), np.full(nb, 4), "tracked", meter_estimated=True)


def build_beat_grid(md: MidiData, cfg: BeatConfig) -> BeatGrid:
    """Beat grid for a parsed file, covering every note."""
    source = cfg.source
    if source == "auto":
        source = "midi" if looks_quantized(md, cfg.auto_quantized_fraction) else "tracked"
    notes = md.notes
    if source == "midi":
        grid = grid_from_midi(md)
    elif source == "tracked":
        grid = track_beats(notes.onset, notes.offset, notes.pitch, cfg)
    else:
        raise ValueError(f"unknown beat source '{cfg.source}'")
    return grid.extended(min(0.0, float(notes.onset.min())), float(notes.offset.max()) + 1.0)
