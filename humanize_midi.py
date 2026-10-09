"""Humanize the note velocities of a MIDI file with a trained model.

    python humanize_midi.py --input input_flat.mid --model models/velocity_xgb/model.json \
        --output output_humanized.mid
    python humanize_midi.py --input song.mid --model models/velocity_xgb/model.json --output out.mid \
        --smoothing --dynamics_scale 1.2 --offset -5

Only note_on velocities change; timing, durations, pedals, tempo and all other
events are copied unchanged. Every non-drum note in the file is treated as part of
one piano texture (that is what the model was trained on).

Beat grid: with the default ``--beat_source auto`` a file whose notes sit on its
own tick grid (DAW/score MIDI) uses its real tempo map and time signatures;
otherwise beats are tracked from the onsets, as during training on MAESTRO.

Residual models predict a deviation from a local baseline. By default that
baseline is taken from the input file's own velocities (``--baseline input``),
so dynamics you drew in (crescendi, a quiet section) are kept and the model adds
note-level shaping; a completely flat input gives a flat baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

import numpy as np

from midi_features import canonical_velocities, featurize_midi, to_note_order
from midi_io import clip_velocities, load_midi, looks_quantized, write_velocities
from model_io import VelocityModel
from postprocess import shape_dynamics, smooth_velocities
from utils import setup_logging

log = logging.getLogger("humanize_midi")


def humanize(args: argparse.Namespace) -> None:
    model = VelocityModel.load(args.model)
    if model.experiment == "performance_conditioned":
        raise SystemExit("this model needs the true performed velocities as input; it cannot humanize MIDI")
    cfg = model.pipeline_config()
    if args.beat_source != "model":
        cfg = dataclasses.replace(cfg, beat=dataclasses.replace(cfg.beat, source=args.beat_source))
    if args.fixed_meter:
        cfg = dataclasses.replace(cfg, beat=dataclasses.replace(cfg.beat, fixed_meter=args.fixed_meter))

    md = load_midi(args.input, include_drums=False)
    if looks_quantized(md, cfg.beat.auto_quantized_fraction) and not cfg.features.quantize:
        log.warning(
            "input looks quantized (grid-aligned onsets), but this model was trained on performance timing. "
            "On score-like input a model trained with --quantize is clearly better "
            "(MAESTRO test: MAE 11.3 vs 13.6); consider models/velocity_xgb_quantized/model.json"
        )
    pf = featurize_midi(md, cfg)
    df = pf.df
    in_vel = canonical_velocities(md, df)
    raw = model.predict_raw(df)
    if model.target_mode == "residual":
        base = model.inference_baseline(in_vel, df["group_id"].to_numpy(), args.baseline, args.base_velocity)
        vel = model.to_velocity(raw, base)
    else:
        vel = raw

    if args.smoothing:
        vel = smooth_velocities(vel, df["group_id"].to_numpy(), args.smoothing_alpha, args.smoothing_strength,
                                args.accent_threshold)
    vel = shape_dynamics(vel, args.dynamics_scale, args.offset)
    vel = args.mix * vel + (1.0 - args.mix) * in_vel
    out_vel = clip_velocities(vel)
    write_velocities(md, to_note_order(out_vel, df), args.output)

    g = pf.grid
    log.info("%d notes; beat grid: %s, meter %s, median tempo %.0f bpm", len(df), g.source, g.meter_label,
             np.median(g.tempo_bpm))
    log.info("input velocity  mean %.1f sd %.1f range %d-%d", in_vel.mean(), in_vel.std(), in_vel.min(), in_vel.max())
    log.info("output velocity mean %.1f sd %.1f range %d-%d", out_vel.mean(), out_vel.std(), out_vel.min(), out_vel.max())
    log.info("wrote %s", args.output)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True, help="model.json from train_xgboost.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--beat_source", default="auto", choices=["auto", "midi", "tracked", "model"],
                    help="'model' = whatever the training data used (tracked for MAESTRO)")
    ap.add_argument("--fixed_meter", type=int, default=None, help="beats per bar when tracking (e.g. 3)")
    ap.add_argument("--smoothing", action="store_true", help="smooth group-to-group jitter (keeps voicing/accents)")
    ap.add_argument("--smoothing_alpha", type=float, default=0.35)
    ap.add_argument("--smoothing_strength", type=float, default=0.5)
    ap.add_argument("--accent_threshold", type=float, default=8.0)
    ap.add_argument("--dynamics_scale", type=float, default=1.0, help=">1 widens, <1 narrows the dynamic range")
    ap.add_argument("--offset", type=float, default=0.0, help="shift all velocities by this amount")
    ap.add_argument("--mix", type=float, default=1.0, help="1 = model only, 0 = keep input velocities")
    ap.add_argument("--baseline", default="input", choices=["input", "constant"], help="residual models only")
    ap.add_argument("--base_velocity", type=float, default=64.0, help="with --baseline constant")
    args = ap.parse_args()
    setup_logging()
    if not 0.0 <= args.mix <= 1.0:
        ap.error("--mix must be between 0 and 1")
    humanize(args)


if __name__ == "__main__":
    main()
