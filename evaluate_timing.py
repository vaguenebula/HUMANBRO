"""Evaluate the timing model: calibration, sharpness, point accuracy, realism of sampled timing,
plots, and MIDI files to listen to.

    python evaluate_timing.py --model models/timing_xgb/model.json
    python evaluate_timing.py --model models/timing_xgb/model.json --split validation --n_midi 4 --temperature 0.8

Writes to <model dir>/eval_<split>/: metrics.json, summary.md, per_file_metrics.csv, plots/*.png, midi/*.mid

Listening test, per selected performance. Velocities, durations and pedalling are the
performer's in every file, so only onset timing differs:
    __1_original.mid         the performance as recorded
    __2_deadpan.mid          the reconstructed score on its smooth timeline, no micro-timing
    __3_median.mid           + each note's median predicted offset (deterministic)
    __4_sampled.mid          + one correlated draw from the predicted distributions
    __5_random_iid.mid       + independent draws from the unconditional offset distribution
                               (what a plain "random humanize" knob does), for comparison
    __6_true_offsets.mid     + the performer's true offsets: the ceiling for this target
                               (the original minus tempo changes slower than the reference beat)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

from config import load_config, add_config_args
from evaluate import INK_2, SERIES, SURFACE, _save, plt, slug, style_matplotlib
from humanize_timing import retime
from midi_io import load_midi, write_timing
from model_io import TimingModel
from timing_features import featurize_performance
from timing_metrics import distribution_metrics, median_metrics, pinball, realism_stats
from timing_sampling import PortableRng, normal_knots, normal_scores, quantile_function, sample_offsets
from utils import format_table, load_feature_table, load_features_meta, save_json, set_seed, setup_logging

log = logging.getLogger("evaluate_timing")


def per_piece_draws(df: pd.DataFrame, q: np.ndarray, model: TimingModel, temperature: float, seed: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """One correlated model draw and one iid unconditional draw per note, piece by piece."""
    rng = PortableRng(seed)
    alphas, knots = model.quantiles, normal_knots(model.quantiles)
    uncond = np.broadcast_to(np.asarray(model.meta["unconditional_quantiles_ms"]), q.shape)
    sampled = np.empty(len(df))
    for _, idx in df.groupby("file_id", sort=False).indices.items():
        part = df.iloc[idx]
        sampled[idx] = sample_offsets(q[idx], alphas, part["group_id"].to_numpy(), part["onset_beat"].to_numpy(),
                                      model.copula, temperature, rng)
    iid = quantile_function(uncond, knots, rng.normals(len(df)))
    return sampled, iid


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(df: pd.DataFrame, q: np.ndarray, sampled: np.ndarray, model: TimingModel, out: Path, split: str) -> None:
    style_matplotlib()
    out.mkdir(parents=True, exist_ok=True)
    alphas = model.quantiles
    y = df["offset_ms"].to_numpy(dtype=np.float64)
    mid = model.median_index()
    uncond = np.asarray(model.meta["unconditional_quantiles_ms"])

    # Calibration: how often the true offset falls below each predicted quantile.
    cov_m = np.mean(y[:, None] <= q, axis=0)
    cov_u = np.mean(y[:, None] <= uncond[None, :], axis=0)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1.0, linestyle="--")
    ax.plot(alphas, cov_m, color=SERIES[0], marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label="model (per-note quantiles)")
    ax.plot(alphas, cov_u, color=SERIES[1], marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label="unconditional (train quantiles)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="quantile level", ylabel="share of notes at or below the prediction")
    ax.set_title(f"Calibration ({split}): on the diagonal = calibrated")
    ax.legend(loc="upper left")
    _save(fig, out / "calibration.png")

    # PIT histogram: flat = the predicted distributions have the right shape and width.
    u = ndtr(normal_scores(q, normal_knots(alphas), y))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(u, bins=np.linspace(0, 1, 21), density=True, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.axhline(1.0, color=INK_2, linewidth=1.0, linestyle="--")
    ax.set(xlim=(0, 1), xlabel="predicted CDF at the true offset (PIT)", ylabel="density")
    ax.set_title("Probability integral transform: flat = calibrated, U = too narrow, hump = too wide")
    _save(fig, out / "pit_histogram.png")

    # Sharpness: per-note 80% interval width vs the one-size-fits-all width.
    lo, hi = int(np.argmin(np.abs(alphas - 0.1))), int(np.argmin(np.abs(alphas - 0.9)))
    width = q[:, hi] - q[:, lo]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(width, bins=np.linspace(0, np.percentile(width, 99.5), 60), color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    uw = uncond[hi] - uncond[lo]
    ax.axvline(uw, color=INK_2, linewidth=1.5, linestyle="--")
    ax.text(uw, ax.get_ylim()[1] * 0.95, f"  unconditional {uw:.0f} ms", color=INK_2, va="top", fontsize=9)
    ax.set(xlabel=f"width of the predicted {alphas[lo]:.0%}-{alphas[hi]:.0%} interval (ms)", ylabel="notes")
    ax.set_title(f"Predicted spread varies by note (median width {np.median(width):.0f} ms)")
    _save(fig, out / "interval_width.png")

    # Melody lead / chord asynchrony by voice position.
    key = df["file_id"].to_numpy().astype(np.int64) * 10_000_000 + df["group_id"].to_numpy()
    s = pd.DataFrame({"k": key, "y": y, "p": q[:, mid], "rank": df["chord_rank_from_top"].to_numpy()})
    g = s.groupby("k")
    s["size"] = g["y"].transform("size")
    s["dy"], s["dp"] = s["y"] - g["y"].transform("mean"), s["p"] - g["p"].transform("mean")
    by = s[(s["size"] >= 3) & (s["rank"] <= 5)].groupby("rank")[["dy", "dp"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color=INK_2, linewidth=1.0)
    ax.plot(by.index, by["dy"], color=SERIES[0], marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label="performed")
    ax.plot(by.index, by["dp"], color=SERIES[1], marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label="model median")
    ax.set(xlabel="voice position in the chord (0 = top note)", ylabel="onset relative to chord mean (ms)")
    ax.set_xticks(by.index)
    ax.set_title("Chord asynchrony by voice (chords of 3+ notes)")
    ax.legend(loc="lower right")
    _save(fig, out / "chord_asynchrony.png")

    # Mean offset by position in the beat.
    frac = (np.round(df["beat_frac"].to_numpy(dtype=np.float64) * 12) / 12 % 1).round(3)
    pos = pd.DataFrame({"frac": frac, "y": y, "p": q[:, mid]}).groupby("frac").agg(
        n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    pos = pos[pos["n"] >= 2000]
    labels = {0.0: "beat", 0.25: "2nd 16th", 0.333: "1st triplet", 0.5: "8th", 0.667: "2nd triplet", 0.75: "4th 16th"}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color=INK_2, linewidth=1.0)
    ax.plot(pos.index, pos["y"], color=SERIES[0], marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label="performed")
    ax.plot(pos.index, pos["p"], color=SERIES[1], marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label="model median")
    ax.set_xticks(pos.index, [labels.get(f, f"{f:g}") for f in pos.index], fontsize=8)
    ax.set(xlabel="position in the beat", ylabel="mean offset (ms)")
    ax.set_title("Mean offset by metrical position")
    ax.legend(loc="lower left")
    _save(fig, out / "offset_by_position.png")

    # One excerpt: chord-level offsets, median, 80% band, and a sampled take.
    fid = df["file_id"].value_counts().index[len(df["file_id"].unique()) // 2]
    part = df.index[df["file_id"] == fid]
    gid = df.loc[part, "group_id"].to_numpy()
    start = int(np.searchsorted(gid, gid.max() // 3))
    sel = part[start:start + 400]
    ex = pd.DataFrame({"g": df.loc[sel, "group_id"], "b": df.loc[sel, "onset_beat"], "y": y[sel],
                       "m": q[sel, mid], "lo": q[sel, lo], "hi": q[sel, hi], "s": sampled[sel]}).groupby("g").mean()
    ex = ex[ex["b"] <= ex["b"].iloc[0] + 16]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(ex["b"], ex["lo"], ex["hi"], color=SERIES[0], alpha=0.18, linewidth=0, label="model 10-90%")
    ax.plot(ex["b"], ex["m"], color=SERIES[0], label="model median")
    ax.plot(ex["b"], ex["s"], color=SERIES[1], linewidth=1.5, marker="o", markersize=4, label="one sampled take")
    ax.scatter(ex["b"], ex["y"], s=36, color=INK_2, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="performed")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    top = float(np.nanmax(ex[["y", "hi", "s"]].to_numpy()))
    bottom = float(np.nanmin(ex[["y", "lo", "s"]].to_numpy()))
    ax.set_ylim(bottom - 5, top + 0.35 * (top - bottom))  # headroom for the legend
    ax.set(xlabel="score position (beats)", ylabel="chord onset offset (ms)")
    ax.set_title(f"16 beats of test performance {fid}: chord-level timing")
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    _save(fig, out / "excerpt.png")

    imp = pd.Series(model.booster.get_score(importance_type="gain")).sort_values(ascending=False).head(25)[::-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.7)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="average gain per split (all quantiles)")
    ax.set_title("Top 25 features by XGBoost gain")
    _save(fig, out / "feature_importance.png")


# ---------------------------------------------------------------------------
# Listening test
# ---------------------------------------------------------------------------


def render_midi(model: TimingModel, files: list[dict[str, Any]], maestro_dir: Path, out: Path, temperature: float,
                seed: int) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    pcfg = model.pipeline_config()
    alphas, knots = model.quantiles, normal_knots(model.quantiles)
    uncond = np.asarray(model.meta["unconditional_quantiles_ms"])
    rng = PortableRng(seed)
    rows = []
    for info in files:
        md = load_midi(maestro_dir / info["midi_filename"])
        piece = featurize_performance(md, pcfg)
        df = piece.df
        q = model.predict_quantiles(df)
        true = df["offset_ms"].to_numpy(dtype=np.float64)
        takes = {
            "2_deadpan": np.zeros(len(df)),
            "3_median": q[:, model.median_index()],
            "4_sampled": sample_offsets(q, alphas, df["group_id"].to_numpy(), df["onset_beat"].to_numpy(),
                                        model.copula, temperature, rng),
            "5_random_iid": quantile_function(np.broadcast_to(uncond, q.shape), knots, rng.normals(len(df))),
            "6_true_offsets": true,
        }
        stem = f"{info['file_id']:04d}_{slug(info['canonical_composer'].split()[-1], 15)}_{slug(info['canonical_title'])}"
        md.midi.save(str(out / f"{stem}__1_original.mid"))
        for tag, ms in takes.items():
            on_tick, off_tick, warp = retime(md, piece, np.clip(ms, -200, 200))
            write_timing(md, on_tick, off_tick, out / f"{stem}__{tag}.mid", warp)
        row = {"file": stem, "notes": len(df), "grid": f"{piece.grid.meter_label}, ~{np.median(piece.grid.tempo_bpm):.0f} bpm",
               "true_sd_ms": float(true.std())}
        for tag in ("3_median", "4_sampled", "5_random_iid"):
            row[f"{tag[2:]}_sd_ms"] = float(takes[tag].std())
        row["median_mae_ms"] = float(np.mean(np.abs(takes["3_median"] - true)))
        row["zero_mae_ms"] = float(np.mean(np.abs(true)))
        rows.append(row)
        log.info("%s: true sd %.1f ms; median MAE %.1f ms (deadpan %.1f)", stem, row["true_sd_ms"],
                 row["median_mae_ms"], row["zero_mae_ms"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/timing_xgb/model.json")
    ap.add_argument("--data", default=None, help="timing feature cache (default: the one the model was trained on)")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--output_dir", default=None, help="default: <model dir>/eval_<split>")
    ap.add_argument("--maestro_dir", default=None)
    ap.add_argument("--n_midi", type=int, default=5, help="performances to render for listening")
    ap.add_argument("--temperature", type=float, default=1.0, help="spread of the sampled takes")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--no_plots", action="store_true")
    add_config_args(ap)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    seed = cfg.train.seed
    set_seed(seed)

    model = TimingModel.load(args.model)
    data = Path(args.data or model.meta["features_data"])
    fmeta = load_features_meta(data)
    out = Path(args.output_dir or Path(args.model).parent / f"eval_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    files = [f for f in fmeta["files"] if f["split"] == args.split]
    rng = np.random.default_rng(seed)
    if args.max_files and len(files) > args.max_files:
        files = [files[i] for i in sorted(rng.choice(len(files), args.max_files, replace=False))]
    cols = list(dict.fromkeys(["file_id", "note_idx", "group_id", "offset_ms", "onset_beat", "beat_frac",
                               "is_chord_top", "chord_rank_from_top", *model.feature_columns]))
    df = load_feature_table(data, cols, [args.split], [f["file_id"] for f in files])
    log.info("%s split: %d performances, %d notes", args.split, df["file_id"].nunique(), len(df))

    alphas = model.quantiles
    q = model.predict_quantiles(df)
    y = df["offset_ms"].to_numpy(dtype=np.float64)
    f, g, top = df["file_id"].to_numpy(), df["group_id"].to_numpy(), df["is_chord_top"].to_numpy()
    uncond = np.asarray(model.meta["unconditional_quantiles_ms"])
    mid = model.median_index()
    dist = {"model": distribution_metrics(y, q, alphas), "unconditional": distribution_metrics(y, uncond, alphas)}
    cop = model.copula
    if cop.calib_u:  # the quantiles the sampler effectively draws from (recalibration fitted on validation)
        knots = normal_knots(alphas)
        z_levels = np.interp(alphas, cop.calib_u, cop.calib_z)
        q_cal = np.stack([quantile_function(q, knots, np.full(len(q), z)) for z in z_levels], axis=1)
        dist["recalibrated"] = distribution_metrics(y, q_cal, alphas)
    if "median_residual_quantiles_ms" in model.meta:  # model median + one fixed spread for every note
        q_homo = q[:, mid:mid + 1] + np.asarray(model.meta["median_residual_quantiles_ms"])[None, :]
        dist["homoscedastic"] = distribution_metrics(y, q_homo, alphas)
    med = median_metrics(y, q[:, mid], f, g, top)
    zero = median_metrics(y, np.zeros(len(y)), f, g, top)
    sampled, iid = per_piece_draws(df, q, model, args.temperature, seed)
    realism = {name: realism_stats(v, f, g) for name, v in
               (("performed", y), ("model median", q[:, mid]), (f"model sampled (T={args.temperature:g})", sampled),
                ("random iid (unconditional)", iid))}

    labels = {"model": "model (per-note quantiles)", "recalibrated": "model, recalibrated on validation",
              "homoscedastic": "model median + fixed spread", "unconditional": "unconditional (train quantiles)"}
    keys = [k for k in ("model", "recalibrated", "homoscedastic", "unconditional") if k in dist]
    rows = [{"model": labels[k], "mean_pinball": dist[k]["mean_pinball"]} for k in keys]
    for r, key in zip(rows, keys):
        for iv in dist[key]["intervals"]:
            r[f"cov_{iv['interval']}"] = iv["coverage"]
            r[f"width_{iv['interval']}"] = iv["mean_width_ms"]
    dist_cols = ["model", "mean_pinball"] + [c for c in rows[0] if c.startswith(("cov_", "width_"))]
    dist_table = format_table(rows, dist_cols, "{:.3f}")
    calib_table = format_table(
        [{"level": a, **{k: dist[k]["calibration"][str(a)] for k in keys}} for a in alphas], ["level", *keys])
    point_rows = []
    for name, m in (("model median", med), ("zero offset (deadpan)", zero)):
        point_rows.append({"prediction": name, "mae_ms": m["all"]["mae"], "rmse_ms": m["all"]["rmse"],
                           "r": m["all"]["pearson"], "within_piece_r": m["all"]["within_piece_pearson"],
                           "event_mae_ms": m["event"]["mae"], "event_r": m["event"]["pearson"],
                           "async_mae_ms": m["asynchrony"]["mae"], "async_r": m["asynchrony"]["pearson"]})
    point_table = format_table(point_rows, list(point_rows[0]), "{:.3f}")
    real_table = format_table([{"offsets": k, **v} for k, v in realism.items()],
                              ["offsets", "sd_ms", "event_sd_ms", "event_lag1_r", "asynchrony_sd_ms"], "{:.2f}")
    lead = med["top_lead_ms"]

    print(f"\nDistribution on the {args.split} split ({len(y):,} notes; offsets in ms; lower pinball is better)")
    print(dist_table)
    print("\nCalibration: share of notes at or below each predicted quantile (ideal = level)")
    print(calib_table)
    print("\nPoint prediction (median); event = chord-level timing, async = within-chord asynchrony")
    print(point_table)
    print(f"\nTop-note lead within chords: performed {lead['true']:.2f} ms, model median {lead['pred']:.2f} ms")
    print(f"\nRealism of generated timing (copula rho {cop.rho:.2f}, correlation length {cop.ell:.2f} beats)")
    print(real_table)

    pb_note = pinball(y, q, alphas)  # per-level means for the whole split
    per = pd.DataFrame({"file_id": f, "abs_err": np.abs(q[:, mid] - y), "abs_y": np.abs(y),
                        "in80": (y >= q[:, int(np.argmin(np.abs(alphas - 0.1)))]) & (y <= q[:, int(np.argmin(np.abs(alphas - 0.9)))])})
    per_file = per.groupby("file_id").agg(notes=("abs_err", "size"), median_mae_ms=("abs_err", "mean"),
                                          zero_mae_ms=("abs_y", "mean"), coverage_80=("in80", "mean")).reset_index()
    names = {x["file_id"]: (x["canonical_composer"], x["canonical_title"]) for x in fmeta["files"]}
    per_file["composer"] = per_file["file_id"].map(lambda i: names[i][0])
    per_file["title"] = per_file["file_id"].map(lambda i: names[i][1])
    per_file.sort_values("median_mae_ms").to_csv(out / "per_file_metrics.csv", index=False)

    if not args.no_plots:
        make_plots(df, q, sampled, model, out / "plots", args.split)
        log.info("plots written to %s", out / "plots")

    listen: list[dict[str, Any]] = []
    if args.n_midi > 0:
        maestro_dir = Path(args.maestro_dir or fmeta["maestro_dir"])
        chosen = [files[i] for i in sorted(rng.choice(len(files), min(args.n_midi, len(files)), replace=False))]
        listen = render_midi(model, chosen, maestro_dir, out / "midi", args.temperature, seed)
        print("\nListening-test files (offsets in ms)")
        print(format_table(listen, list(listen[0]), "{:.1f}"))

    save_json({"split": args.split, "notes": len(y), "distribution": dist, "median": med, "zero_offset": zero,
               "realism": realism, "copula": cop.to_dict(), "pinball_per_level": pb_note.tolist(),
               "temperature": args.temperature, "listening_test": listen, "model": str(args.model)},
              out / "metrics.json")
    parts = [f"# Timing evaluation: {args.model} on {args.split}\n", "## Distribution\n", dist_table,
             "\n## Calibration\n", calib_table, "\n## Point prediction (median)\n", point_table,
             f"\nTop-note lead: performed {lead['true']:.2f} ms, model {lead['pred']:.2f} ms\n",
             "\n## Realism of generated timing\n", real_table]
    if listen:
        parts += ["\n## Listening test\n", format_table(listen, list(listen[0]), "{:.1f}")]
    (out / "summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    log.info("results written to %s", out)


if __name__ == "__main__":
    main()
