"""MIDI reading/writing on top of mido.

Design notes
------------
* Timing: every tempo change in every track is honoured when converting ticks
  to seconds (``TempoMap``). Time-signature events are kept, not discarded.
* Notes: a note starts at a note_on with velocity > 0 and ends at the matching
  note_off (or note_on with velocity 0). A re-strike of a key that is still
  down closes the previous instance at the re-strike tick.
* Sustain (CC64) and soft (CC67) pedals are *not* applied to note durations:
  durations are key-down durations, the closest thing a performance has to
  notated length. Pedals are performance data and are not used as features.
  They are preserved untouched when a file is re-written.
* Every note remembers which message (track index, message index) started it,
  and which message ended it, so writing new velocities or new timing edits
  exactly those messages and leaves every other event (pedals, tempo, program
  changes) as it was.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mido
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_TEMPO = 500_000  # microseconds per quarter note (120 BPM), the MIDI default
DRUM_CHANNEL = 9
MAX_GRID_BEATS = 500_000  # sanity cap for corrupt files with absurd lengths


class MidiParseError(RuntimeError):
    """A file that cannot be turned into a usable note list."""


@dataclass
class TempoMap:
    ticks_per_beat: int
    change_ticks: np.ndarray  # int64, sorted, first entry is 0
    tempos: np.ndarray  # microseconds per quarter note from change_ticks[i] onward
    change_seconds: np.ndarray  # absolute time of each change

    @classmethod
    def from_changes(cls, ticks_per_beat: int, changes: list[tuple[int, int]]) -> TempoMap:
        by_tick: dict[int, int] = {}
        # Stable sort: for several tempo events on one tick, the last one in file order wins.
        for tick, tempo in sorted(changes, key=lambda c: c[0]):
            if tempo > 0:
                by_tick[tick] = tempo
        by_tick.setdefault(0, DEFAULT_TEMPO)
        ticks = np.array(sorted(by_tick), dtype=np.int64)
        tempos = np.array([by_tick[t] for t in ticks], dtype=np.float64)
        sec_per_tick = tempos / 1e6 / ticks_per_beat
        seconds = np.concatenate([[0.0], np.cumsum(np.diff(ticks) * sec_per_tick[:-1])])
        return cls(ticks_per_beat, ticks, tempos, seconds)

    def _segment(self, ticks: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.change_ticks, ticks, side="right") - 1
        return np.clip(idx, 0, len(self.change_ticks) - 1)

    def ticks_to_seconds(self, ticks: np.ndarray | float) -> np.ndarray:
        ticks = np.asarray(ticks, dtype=np.float64)
        i = self._segment(ticks)
        return self.change_seconds[i] + (ticks - self.change_ticks[i]) * self.tempos[i] / 1e6 / self.ticks_per_beat

    def seconds_to_ticks(self, seconds: np.ndarray | float) -> np.ndarray:
        """Inverse of ``ticks_to_seconds`` (fractional ticks; times before 0 extrapolate)."""
        s = np.asarray(seconds, dtype=np.float64)
        i = np.clip(np.searchsorted(self.change_seconds, s, side="right") - 1, 0, len(self.change_ticks) - 1)
        return self.change_ticks[i] + (s - self.change_seconds[i]) * 1e6 * self.ticks_per_beat / self.tempos[i]

    @property
    def is_constant(self) -> bool:
        return len(np.unique(self.tempos)) == 1


@dataclass
class NoteArray:
    onset: np.ndarray  # seconds
    offset: np.ndarray  # seconds (key release, pedal ignored)
    onset_tick: np.ndarray
    offset_tick: np.ndarray
    pitch: np.ndarray
    velocity: np.ndarray  # the target; never passed to feature extraction
    channel: np.ndarray
    track: np.ndarray  # track index of the note_on message
    msg_index: np.ndarray  # index of the note_on message within that track
    off_msg_index: np.ndarray  # index of the message that ended it (-1: re-strike or end of track)

    def __len__(self) -> int:
        return len(self.pitch)


@dataclass
class MidiData:
    path: Path | None
    midi: mido.MidiFile
    tempo_map: TempoMap
    time_signatures: list[tuple[int, int, int]]  # (tick, numerator, denominator), sorted, starts at 0
    notes: NoteArray
    end_tick: int

    @property
    def ticks_per_beat(self) -> int:
        return self.tempo_map.ticks_per_beat

    @property
    def end_time(self) -> float:
        return float(self.tempo_map.ticks_to_seconds(self.end_tick))


def load_midi(path: str | Path, include_drums: bool = False) -> MidiData:
    try:
        mf = mido.MidiFile(str(path), clip=True)
    except Exception as exc:  # mido raises a variety of errors for malformed files
        raise MidiParseError(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc
    return parse_midi(mf, Path(path), include_drums=include_drums)


def parse_midi(mf: mido.MidiFile, path: Path | None = None, include_drums: bool = False) -> MidiData:
    if mf.type == 2:
        raise MidiParseError("type-2 (asynchronous tracks) MIDI files are not supported")
    tpb = mf.ticks_per_beat
    if not tpb or tpb <= 0:
        raise MidiParseError("SMPTE / invalid time division is not supported")

    tempo_changes: list[tuple[int, int]] = []
    time_sigs: list[tuple[int, int, int]] = []
    on_tick: list[int] = []
    off_tick: list[int] = []
    pitch: list[int] = []
    vel: list[int] = []
    chan: list[int] = []
    trk: list[int] = []
    midx: list[int] = []
    off_midx: list[int] = []
    end_tick = 0

    for ti, track in enumerate(mf.tracks):
        tick = 0
        open_notes: dict[tuple[int, int], int] = {}
        for mi, msg in enumerate(track):
            tick += msg.time
            mtype = msg.type
            if mtype == "note_on" or mtype == "note_off":
                if msg.channel == DRUM_CHANNEL and not include_drums:
                    continue
                key = (msg.channel, msg.note)
                if mtype == "note_on" and msg.velocity > 0:
                    prev = open_notes.pop(key, None)
                    if prev is not None:  # re-strike while the key is still down
                        off_tick[prev] = tick
                    open_notes[key] = len(pitch)
                    on_tick.append(tick)
                    off_tick.append(-1)
                    off_midx.append(-1)
                    pitch.append(msg.note)
                    vel.append(msg.velocity)
                    chan.append(msg.channel)
                    trk.append(ti)
                    midx.append(mi)
                else:
                    prev = open_notes.pop(key, None)
                    if prev is not None:
                        off_tick[prev] = tick
                        off_midx[prev] = mi
            elif mtype == "set_tempo":
                tempo_changes.append((tick, msg.tempo))
            elif mtype == "time_signature":
                if msg.numerator > 0 and msg.denominator > 0:
                    time_sigs.append((tick, msg.numerator, msg.denominator))
        for idx in open_notes.values():  # keys never released: end at end of track
            off_tick[idx] = tick
        end_tick = max(end_tick, tick)

    if not pitch:
        raise MidiParseError("no notes found")

    tempo_map = TempoMap.from_changes(tpb, tempo_changes)

    ts_by_tick: dict[int, tuple[int, int]] = {}
    for tick, num, den in sorted(time_sigs, key=lambda t: t[0]):
        ts_by_tick[tick] = (num, den)
    ts_by_tick.setdefault(0, (4, 4))  # MIDI default when no time signature is given
    time_signatures = [(t, *ts_by_tick[t]) for t in sorted(ts_by_tick)]

    on_t = np.asarray(on_tick, dtype=np.int64)
    off_t = np.maximum(np.asarray(off_tick, dtype=np.int64), on_t)
    p = np.asarray(pitch, dtype=np.int16)
    order = np.lexsort((p, on_t))
    notes = NoteArray(
        onset=tempo_map.ticks_to_seconds(on_t[order]),
        offset=tempo_map.ticks_to_seconds(off_t[order]),
        onset_tick=on_t[order],
        offset_tick=off_t[order],
        pitch=p[order],
        velocity=np.asarray(vel, dtype=np.int16)[order],
        channel=np.asarray(chan, dtype=np.int8)[order],
        track=np.asarray(trk, dtype=np.int32)[order],
        msg_index=np.asarray(midx, dtype=np.int32)[order],
        off_msg_index=np.asarray(off_midx, dtype=np.int32)[order],
    )
    end_tick = max(end_tick, int(off_t.max()))
    return MidiData(path, mf, tempo_map, time_signatures, notes, end_tick)


def clip_velocities(velocities: np.ndarray) -> np.ndarray:
    """Round to integers and clip to the valid note-on range 1..127 (0 would mean note-off)."""
    v = np.asarray(velocities, dtype=np.float64)
    v = np.where(np.isfinite(v), v, 64.0)
    return np.clip(np.rint(v), 1, 127).astype(np.int64)


def with_velocities(md: MidiData, velocities: np.ndarray) -> mido.MidiFile:
    """Copy of the original file where each note_on gets a new velocity.

    ``velocities`` is indexed like ``md.notes``. Only the note_on messages that
    start notes are replaced; everything else is shared with the original.
    """
    if len(velocities) != len(md.notes):
        raise ValueError(f"expected {len(md.notes)} velocities, got {len(velocities)}")
    v = clip_velocities(velocities)
    src = md.midi
    out = mido.MidiFile(type=src.type, ticks_per_beat=src.ticks_per_beat, charset=src.charset)
    out.tracks = [mido.MidiTrack(list(track)) for track in src.tracks]
    for k in range(len(v)):
        track = out.tracks[int(md.notes.track[k])]
        mi = int(md.notes.msg_index[k])
        track[mi] = track[mi].copy(velocity=int(v[k]))
    return out


def write_velocities(md: MidiData, velocities: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with_velocities(md, velocities).save(str(path))
    return path


def _same_key_consistent(notes: NoteArray, on: np.ndarray, off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per key (track, channel, pitch), in original order: strictly increasing onsets, and
    every note released no later than the next strike of its key (else that release would
    cut the next note short)."""
    n = len(on)
    key = (notes.track.astype(np.int64) * 16 + notes.channel) * 128 + notes.pitch
    order = np.lexsort((np.arange(n), key))  # md.notes is time-sorted, so this is per-key time order
    k = key[order]
    new_seg = np.concatenate([[True], k[1:] != k[:-1]])
    seg = np.cumsum(new_seg) - 1
    seg_start = np.flatnonzero(new_seg)
    j = np.arange(n) - seg_start[seg]  # position within the key's sequence
    w = on[order] - j
    span = float(np.ptp(w)) + 1.0
    w = np.maximum.accumulate(w + seg * span) - seg * span  # running max restarted per key
    on_sorted = w + j
    same_key_next = np.concatenate([~new_seg[1:], [False]])
    nxt = np.where(same_key_next, np.concatenate([on_sorted[1:], [np.inf]]), np.inf)
    off_sorted = np.clip(off[order], on_sorted + 1, nxt)
    on_out, off_out = np.empty(n), np.empty(n)
    on_out[order], off_out[order] = on_sorted, off_sorted
    return on_out.astype(np.int64), off_out.astype(np.int64)


def with_timing(
    md: MidiData,
    onset_tick: np.ndarray,
    offset_tick: np.ndarray,
    warp: Callable[[np.ndarray], np.ndarray] | None = None,
) -> mido.MidiFile:
    """Copy of the file with every note moved to new onset/offset ticks.

    ``onset_tick``/``offset_tick`` are indexed like ``md.notes`` (fractional ticks are
    rounded). Every other event keeps its tick, or is mapped through ``warp`` (ticks ->
    ticks, vectorised) when the whole timeline is re-timed, so pedals follow the notes.
    Same-key notes keep their order, and a release never cuts the next strike short.
    Velocities, channels and all other message contents are unchanged.
    """
    notes = md.notes
    n = len(notes)
    if len(onset_tick) != n or len(offset_tick) != n:
        raise ValueError(f"expected {n} onsets and offsets")
    on = np.maximum(np.rint(np.asarray(onset_tick, dtype=np.float64)), 0)
    off = np.rint(np.asarray(offset_tick, dtype=np.float64))
    on, off = _same_key_consistent(notes, on, np.maximum(off, on + 1))

    src = md.midi
    out = mido.MidiFile(type=src.type, ticks_per_beat=src.ticks_per_beat, charset=src.charset)
    for ti, track in enumerate(src.tracks):
        msgs = list(track)
        if not msgs:
            out.tracks.append(mido.MidiTrack())
            continue
        abs_t = np.cumsum([m.time for m in msgs]).astype(np.float64)
        new_t = np.rint(warp(abs_t)) if warp is not None else abs_t.copy()
        new_t = np.maximum(new_t, 0)
        # Order of events sharing a tick: meta, note releases, other channel events, note starts.
        cls = np.array([0 if m.is_meta else 1 if m.type == "note_off" or (m.type == "note_on" and m.velocity == 0)
                        else 3 if m.type == "note_on" else 2 for m in msgs])
        mine = np.flatnonzero(notes.track == ti)
        new_t[notes.msg_index[mine]] = on[mine]
        ended = mine[notes.off_msg_index[mine] >= 0]
        new_t[notes.off_msg_index[ended]] = off[ended]
        keep = np.array([m.type != "end_of_track" for m in msgs])
        idx = np.flatnonzero(keep)
        idx = idx[np.lexsort((idx, cls[idx], new_t[idx]))]
        new_track = mido.MidiTrack()
        last = 0
        for i in idx:
            t = int(new_t[i])
            new_track.append(msgs[i].copy(time=t - last))
            last = t
        end = int(max(new_t[~keep].max() if (~keep).any() else last, last))
        new_track.append(mido.MetaMessage("end_of_track", time=end - last))
        out.tracks.append(new_track)
    return out


def write_timing(
    md: MidiData,
    onset_tick: np.ndarray,
    offset_tick: np.ndarray,
    path: str | Path,
    warp: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with_timing(md, onset_tick, offset_tick, warp).save(str(path))
    return path


def flatten_velocities(md: MidiData, value: int = 64) -> MidiData:
    """A deadpan version of the file: every note_on velocity set to ``value``."""
    flat = with_velocities(md, np.full(len(md.notes), value))
    return parse_midi(flat, md.path)


def looks_quantized(md: MidiData, fraction: float = 0.8) -> bool:
    """True if note onsets sit on a 1/16 or 1/12-beat tick grid (i.e. the tempo map is real).

    For an unquantized performance with a dummy tempo map only ~6-8% of onsets
    land within the tolerance by chance; quantized/score MIDI is near 100%.
    """
    ticks = md.notes.onset_tick.astype(np.float64)
    if len(ticks) < 8:
        return False
    tpb = md.ticks_per_beat
    tol = max(1.0, tpb / 240.0)
    dist = np.full(len(ticks), np.inf)
    for step in (tpb / 4.0, tpb / 3.0):
        dist = np.minimum(dist, np.abs(ticks - np.rint(ticks / step) * step))
    return float(np.mean(dist <= tol)) >= fraction
