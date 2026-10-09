"""Humanize the note timing of a MIDI file with the quantile timing model.

    python humanize_timing.py --input clip.mid --model models/timing_xgb/model.json --output clip_humanized.mid
    python humanize_timing.py ... --mode median                 # deterministic: every note gets its median offset
    python humanize_timing.py ... --temperature 0.7 --seed 3    # a tighter random take; change --seed for another

Each note gets an onset offset drawn from the distribution the model predicts for it
(quantile regression). Draws are correlated the way real performances are (timing_sampling.py):
the notes of a chord move largely together, and timing drifts smoothly over about a beat
instead of jittering note by note. Durations are kept, so note releases move with their onsets.

Input:
* quantized DAW/score MIDI (the usual case): its own tempo map and time signatures are the
  grid, and offsets are added to the notes' positions. Tempo, pedals and all other events
  stay where they are.
* a played performance (not on its tick grid): beats are tracked, the score is reconstructed
  as during training, and the file is re-timed on that score with new micro-timing (the
  performer's own timing is replaced). Pedals and other events follow the new timeline.

Velocities are never changed; run humanize_midi.py for those (before or after, in either order).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Callable

import numpy as np

from midi_features import to_note_order
from midi_io import MidiData, load_midi, write_timing
from model_io import TimingModel
from timing_features import TimingPiece, featurize_for_humanizing
from timing_sampling import CopulaParams, PortableRng, normal_knots, quantile_function, sample_offsets
from utils import setup_logging

log = logging.getLogger("humanize_timing")


def predict_offsets_ms(model: TimingModel, piece: TimingPiece, mode: str = "sample", temperature: float = 1.0,
                       seed: int | None = None, copula: CopulaParams | None = None) -> np.ndarray:
    """One offset (ms) per note in canonical order."""
    q = model.predict_quantiles(piece.df)
    if mode == "median":
        return quantile_function(q, normal_knots(model.quantiles), np.zeros(len(q)))
    rng = PortableRng(PortableRng.new_seed() if seed is None else seed)
    df = piece.df
    return sample_offsets(q, model.quantiles, df["group_id"].to_numpy(), df["onset_beat"].to_numpy(),
                          copula or model.copula, temperature, rng)


def retime(md: MidiData, piece: TimingPiece, offsets_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, Callable | None]:
    """New (onset, release) ticks per note in ``md.notes`` order, plus the warp for other events.

    Each note starts at its score time plus its offset and keeps its duration. For a
    re-timed performance, non-note events (pedals) are moved onto the score timeline too.
    """
    base_on = to_note_order(piece.df["onset_s"].to_numpy(), piece.df)
    new_on = base_on + to_note_order(offsets_ms, piece.df) / 1000.0
    tmap = md.tempo_map
    on_tick = tmap.seconds_to_ticks(new_on)
    off_tick = tmap.seconds_to_ticks(new_on + (md.notes.offset - md.notes.onset))
    warp = None
    if piece.score is not None:
        sc = piece.score
        warp = lambda ticks: tmap.seconds_to_ticks(sc.warp_seconds(tmap.ticks_to_seconds(ticks)))  # noqa: E731
    return on_tick, off_tick, warp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="models/timing_xgb/model.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["sample", "median"], default="sample",
                    help="sample: a correlated draw from each note's predicted distribution; median: deterministic")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="spread of the draws: 0 = medians, 1 = as wide as the data (default), >1 exaggerates")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every final offset (0.5 = half as loose)")
    ap.add_argument("--max_offset_ms", type=float, default=150.0, help="clamp offsets to +/- this")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (default: a different take every run; the seed used is printed). "
                         "The C++ runtime gives the same take for the same seed")
    ap.add_argument("--chord_coupling", type=float, default=None,
                    help="override the fitted within-chord correlation rho (0..1; 1 = chords move as one)")
    ap.add_argument("--correlation_beats", type=float, default=None,
                    help="override the fitted correlation length (beats) of the timing drift")
    ap.add_argument("--beat_source", choices=["auto", "midi", "tracked"], default="auto",
                    help="auto: the file's tempo map if its notes sit on the tick grid, else track beats")
    ap.add_argument("--snap", action="store_true", help="quantize chords to the grid first (nearly-quantized input)")
    args = ap.parse_args()
    setup_logging()
    if args.temperature < 0 or args.scale < 0:
        ap.error("--temperature and --scale must be >= 0")
    humanize(args)


def humanize(args: argparse.Namespace) -> np.ndarray:
    """The CLI's work (also called by tests/test_cpp_parity.py). Returns the offsets in ms (canonical order)."""
    model = TimingModel.load(args.model)
    cop = model.copula
    if args.chord_coupling is not None:
        cop = replace(cop, rho=float(np.clip(args.chord_coupling, 0.0, 1.0)))
    if args.correlation_beats is not None:
        cop = replace(cop, ell=max(args.correlation_beats, 1e-3))

    md = load_midi(args.input, include_drums=False)
    piece = featurize_for_humanizing(md, model.pipeline_config(), args.beat_source, args.snap)
    seed = PortableRng.new_seed() if args.seed is None else args.seed
    ms = np.clip(args.scale * predict_offsets_ms(model, piece, args.mode, args.temperature, seed, cop),
                 -args.max_offset_ms, args.max_offset_ms)
    on_tick, off_tick, warp = retime(md, piece, ms)
    write_timing(md, on_tick, off_tick, args.output, warp)

    g = piece.grid
    log.info("%d notes; grid: %s, meter %s, median tempo %.0f bpm%s", len(ms), g.source, g.meter_label,
             np.median(g.tempo_bpm), "" if piece.score is None else " (performance re-timed on its reconstructed score)")
    how = "median" if args.mode == "median" else (f"sampled with --seed {seed}, temperature {args.temperature:g}, "
                                                   f"chord coupling {cop.rho:.2f}, correlation {cop.ell:.2f} beats")
    log.info("offsets (%s, scale %g): sd %.1f ms, 90%% within +/-%.0f ms", how, args.scale, ms.std(),
             np.percentile(np.abs(ms), 90))
    log.info("wrote %s", args.output)
    return ms


if __name__ == "__main__":
    main()
