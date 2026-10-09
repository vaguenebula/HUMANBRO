"""Evaluate a trained model: metrics vs baselines, diagnostic plots, and MIDI files to listen to.

    python evaluate.py --model models/velocity_xgb/model.json --data data/features.parquet
    python evaluate.py --model models/velocity_xgb/model.json --split validation --n_midi 8 --smoothing

Writes to <model dir>/eval_<split>/:
    metrics.json, summary.md, per_file_metrics.csv, plots/*.png, midi/*.mid

Listening test (per selected performance):
    __1_original.mid                  the performance as recorded
    __2_flat64.mid                    every velocity set to 64 (deadpan input)
    __3_humanized_from_flat.mid       model output computed from the flat file
    __4_humanized_from_flat_smoothed  (with --smoothing)
    __5_pred_oracle_baseline.mid      (residual models) delta on top of the performance's own baseline

The model's raw output for the flat file and for the original must be identical:
the features never see velocity. This is checked for every file and reported.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config import add_config_args, config_from_dict, load_config  # noqa: E402
from midi_features import canonical_velocities, featurize_midi, to_note_order  # noqa: E402
from midi_io import clip_velocities, flatten_velocities, load_midi, write_velocities  # noqa: E402
from model_io import VelocityModel  # noqa: E402
from postprocess import apply_smoothing  # noqa: E402
from utils import (  # noqa: E402
    format_table,
    load_feature_table,
    load_features_meta,
    regression_metrics,
    save_json,
    set_seed,
    setup_logging,
    within_piece_pearson,
)

log = logging.getLogger("evaluate")

# Chart styling (validated reference palette: categorical slots 1-2, blue sequential ramp).
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#c9c8c2",
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "font.size": 10,
        }
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_row(name: str, y: np.ndarray, p: np.ndarray, file_ids: np.ndarray) -> dict[str, Any]:
    m = regression_metrics(y, p)
    return {"model": name, **m, "within_piece_r": within_piece_pearson(y, p, file_ids)}


def piece_mean(values: np.ndarray, file_ids: np.ndarray) -> np.ndarray:
    s = pd.Series(values).groupby(file_ids).transform("mean")
    return s.to_numpy()


def compute_metrics(df: pd.DataFrame, pred_vel: np.ndarray, model: VelocityModel, smoothed: np.ndarray | None) -> list[dict[str, Any]]:
    y = df["velocity"].to_numpy(dtype=np.float64)
    f = df["file_id"].to_numpy()
    stats = model.meta["train_stats"]
    per_pitch = np.asarray(stats["per_pitch_mean"])
    residual = model.target_mode == "residual"
    label = "model (oracle local baseline + predicted delta)" if residual else "model"
    rows = [metric_row(label, y, pred_vel, f)]
    if smoothed is not None:
        rows.append(metric_row(label + " + smoothing", y, smoothed, f))
    rows += [
        metric_row("baseline: global mean (train)", y, np.full(len(y), stats["global_mean"]), f),
        metric_row("baseline: global median (train)", y, np.full(len(y), stats["global_median"]), f),
        metric_row("baseline: per-pitch mean (train)", y, per_pitch[df["pitch"].to_numpy().astype(int)], f),
        # The rows below look at the target and are NOT achievable from a score; they
        # bound how much error is due to per-performance loudness the model cannot know.
        metric_row("oracle: performance mean velocity", y, piece_mean(y, f), f),
        metric_row("oracle: local baseline (neighbour median)", y, df["baseline_velocity"].to_numpy(), f),
    ]
    if not residual:
        calibrated = pred_vel - piece_mean(pred_vel, f) + piece_mean(y, f)
        rows.append(metric_row("oracle-calibrated model (true performance mean)", y, np.clip(calibrated, 1, 127), f))
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(df: pd.DataFrame, pred: np.ndarray, model: VelocityModel, out: Path, split: str, shap_samples: int, seed: int) -> None:
    style_matplotlib()
    out.mkdir(parents=True, exist_ok=True)
    y = df["velocity"].to_numpy(dtype=np.float64)
    pitch = df["pitch"].to_numpy().astype(int)
    err = pred - y
    abs_err = np.abs(err)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    hb = ax.hexbin(y, pred, gridsize=64, cmap=BLUE_RAMP, bins="log", mincnt=1, extent=(0, 128, 0, 128), linewidths=0)
    ax.plot([0, 127], [0, 127], color=INK_2, linewidth=1.0, linestyle="--")
    ax.text(118, 122, "y = x", color=INK_2, ha="right", fontsize=9)
    ax.set(xlim=(0, 128), ylim=(0, 128), xlabel="performed velocity", ylabel="predicted velocity")
    ax.set_title(f"Predicted vs. performed velocity ({split}, {len(y):,} notes)")
    fig.colorbar(hb, ax=ax, label="notes per cell (log scale)")
    _save(fig, out / "pred_vs_actual.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=np.arange(-60, 61, 2), color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(0, color=INK_2, linewidth=1.0)
    ax.set(xlabel="prediction error (predicted - performed)", ylabel="notes")
    ax.set_title(f"Residuals: mean {err.mean():+.2f}, sd {err.std():.2f}")
    _save(fig, out / "residual_hist.png")

    per_pitch = np.asarray(model.meta["train_stats"]["per_pitch_mean"])
    base_err = np.abs(per_pitch[pitch] - y)
    by_pitch = pd.DataFrame({"pitch": pitch, "model": abs_err, "base": base_err}).groupby("pitch")
    agg = by_pitch.mean()[by_pitch.size() >= 200]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(agg.index, agg["model"], color=SERIES[0], label="model")
    ax.plot(agg.index, agg["base"], color=SERIES[1], label="per-pitch mean baseline")
    ax.set(xlabel="MIDI pitch", ylabel="mean absolute error", ylim=(0, 1.08 * float(agg.to_numpy().max())))
    ax.set_title("Error by pitch (pitches with at least 200 notes)")
    ax.legend(loc="lower right", ncol=2)
    _save(fig, out / "error_vs_pitch.png")

    bins = np.arange(0, 132, 4)
    centre = (bins[:-1] + bins[1:]) / 2
    which = np.digitize(y, bins) - 1
    counts = np.bincount(which, minlength=len(centre))[: len(centre)]
    ok = counts >= 100
    mae_b = np.bincount(which, weights=abs_err, minlength=len(centre))[: len(centre)] / np.maximum(counts, 1)
    bias_b = np.bincount(which, weights=err, minlength=len(centre))[: len(centre)] / np.maximum(counts, 1)
    gm = model.meta["train_stats"]["global_mean"]
    gm_b = np.bincount(which, weights=np.abs(gm - y), minlength=len(centre))[: len(centre)] / np.maximum(counts, 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(centre[ok], mae_b[ok], color=SERIES[0], label="model")
    a1.plot(centre[ok], gm_b[ok], color=SERIES[1], label="global-mean baseline")
    a1.set(ylabel="mean absolute error")
    a1.set_title("Error by performed velocity (bins of 4, at least 100 notes)")
    a1.legend(loc="upper center")
    a2.axhline(0, color=INK_2, linewidth=1.0)
    a2.plot(centre[ok], bias_b[ok], color=SERIES[0])
    a2.set(xlabel="performed velocity", ylabel="mean signed error (model)")
    a2.set_title("Mean signed error (predicted - performed)", fontsize=10, fontweight="normal")
    _save(fig, out / "error_vs_target_velocity.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    edges = np.arange(0, 129, 2)
    ax.hist(y, bins=edges, histtype="step", color=SERIES[0], linewidth=2, label="performed", density=True)
    ax.hist(np.clip(pred, 1, 127), bins=edges, histtype="step", color=SERIES[1], linewidth=2, label="predicted", density=True)
    ax.set(xlabel="velocity", ylabel="density")
    ax.set_title(f"Velocity distributions (performed sd {y.std():.1f}, predicted sd {pred.std():.1f})")
    ax.legend(loc="upper left")
    _save(fig, out / "velocity_distributions.png")

    imp = pd.Series(model.booster.get_score(importance_type="gain")).sort_values(ascending=False).head(30)[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.7)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="average gain per split")
    ax.set_title("Top 30 features by XGBoost gain")
    _save(fig, out / "feature_importance.png")

    shap_plot(df, model, out, shap_samples, seed)


def shap_plot(df: pd.DataFrame, model: VelocityModel, out: Path, n: int, seed: int) -> None:
    if n <= 0:
        return
    sample = df.sample(min(n, len(df)), random_state=seed)
    x = sample[model.feature_columns].astype(np.float32)
    try:
        import shap

        values = shap.TreeExplainer(model.booster).shap_values(x)
        plt.figure()
        shap.summary_plot(values, x, max_display=30, show=False)
        plt.title("SHAP summary (test sample)")
        plt.savefig(out / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        return
    except ImportError:
        log.info("shap not installed; plotting mean |SHAP| from XGBoost's built-in TreeSHAP instead")
    import xgboost as xgb

    contrib = model.booster.predict(xgb.DMatrix(x), pred_contribs=True)[:, :-1]  # last column = bias
    mean_abs = pd.Series(np.abs(contrib).mean(axis=0), index=model.feature_columns).sort_values(ascending=False)
    top = mean_abs.head(30)[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top.index, top.values, color=SERIES[0], height=0.7)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="mean |SHAP value| (velocity units)")
    ax.set_title(f"Feature impact, TreeSHAP on {len(x):,} notes")
    _save(fig, out / "shap_summary.png")
    mean_abs.to_csv(out / "shap_mean_abs.csv", header=["mean_abs_shap"])


# ---------------------------------------------------------------------------
# Listening test
# ---------------------------------------------------------------------------


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:n]


def musical_eval(
    model: VelocityModel, files: list[dict[str, Any]], maestro_dir: Path, out: Path, smoothing_cfg: Any
) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    pcfg = model.pipeline_config()
    rows = []
    for info in files:
        md = load_midi(maestro_dir / info["midi_filename"])
        flat = flatten_velocities(md, 64)
        pf_orig = featurize_midi(md, pcfg)
        pf_flat = featurize_midi(flat, pcfg)
        raw_orig = model.predict_raw(pf_orig.df)
        raw_flat = model.predict_raw(pf_flat.df)
        same_features = pf_orig.df.equals(pf_flat.df)
        max_diff = float(np.max(np.abs(raw_orig - raw_flat)))
        if not same_features or max_diff > 1e-6:
            log.error("LEAKAGE CHECK FAILED for %s: features equal=%s, max output diff=%.4g",
                      info["midi_filename"], same_features, max_diff)

        group_id = pf_flat.df["group_id"].to_numpy()
        true_c = canonical_velocities(md, pf_orig.df)
        if model.target_mode == "residual":
            base_flat = model.inference_baseline(canonical_velocities(flat, pf_flat.df), group_id, "input")
            v_flat = model.to_velocity(raw_flat, base_flat)
        else:
            v_flat = raw_flat
        stem = f"{info['file_id']:04d}_{slug(info['canonical_composer'].split()[-1], 15)}_{slug(info['canonical_title'])}"

        def write(tag: str, values: np.ndarray, src: Any, feats: pd.DataFrame) -> None:
            write_velocities(src, to_note_order(values, feats), out / f"{stem}__{tag}.mid")

        md.midi.save(str(out / f"{stem}__1_original.mid"))
        flat.midi.save(str(out / f"{stem}__2_flat64.mid"))
        write("3_humanized_from_flat", v_flat, flat, pf_flat.df)
        row = {
            "file": stem,
            "notes": len(true_c),
            "beat_grid": f"{pf_flat.grid.source}, {pf_flat.grid.meter_label}, ~{np.median(pf_flat.grid.tempo_bpm):.0f} bpm",
            "leakage_check": "pass" if same_features and max_diff <= 1e-6 else "FAIL",
            "flat64_mae": float(np.mean(np.abs(64 - true_c))),
        }
        hum = clip_velocities(v_flat)
        m = regression_metrics(true_c, hum)
        row.update(humanized_mae=m["mae"], humanized_r=m["pearson"], true_sd=float(true_c.std()), humanized_sd=float(hum.std()))
        if smoothing_cfg is not None:
            sm = apply_smoothing(v_flat, group_id, smoothing_cfg)
            write("4_humanized_from_flat_smoothed", sm, flat, pf_flat.df)
            ms = regression_metrics(true_c, clip_velocities(sm))
            row.update(smoothed_mae=ms["mae"], smoothed_r=ms["pearson"])
        if model.target_mode == "residual":
            base_orig = model.inference_baseline(true_c, group_id, "input")  # oracle: uses the performance
            write("5_pred_oracle_baseline", model.to_velocity(raw_orig, base_orig), md, pf_orig.df)
        rows.append(row)
        log.info("%s: humanized MAE %.2f (flat64 %.2f), r %.3f, leakage check %s",
                 stem, row["humanized_mae"], row["flat64_mae"], row["humanized_r"], row["leakage_check"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to model.json")
    ap.add_argument("--data", default=None, help="feature cache (default: the one the model was trained on)")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--output_dir", default=None, help="default: <model dir>/eval_<split>")
    ap.add_argument("--maestro_dir", default=None, help="MIDI root for the listening test (default: from the cache)")
    ap.add_argument("--n_midi", type=int, default=5, help="performances to render for listening")
    ap.add_argument("--smoothing", action="store_true", help="also evaluate/write smoothed predictions")
    ap.add_argument("--shap_samples", type=int, default=5000)
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--max_files", type=int, default=None, help="evaluate on a random subset of performances")
    add_config_args(ap)
    args = ap.parse_args()
    setup_logging()

    cfg = load_config(args.config, args.set)
    if args.smoothing:
        cfg = config_from_dict({"smoothing": {"enabled": True}}, base=cfg)
    set_seed(cfg.train.seed)
    model = VelocityModel.load(args.model)
    if model.experiment == "performance_conditioned":
        log.warning("PERFORMANCE-CONDITIONED model: inputs include true neighbour velocities. "
                    "Metrics are an upper-bound analysis, not humanization quality; no MIDI is rendered.")
    data = Path(args.data or model.meta["features_data"])
    fmeta = load_features_meta(data)
    out = Path(args.output_dir or Path(args.model).parent / f"eval_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    files = [f for f in fmeta["files"] if f["split"] == args.split]
    rng = np.random.default_rng(cfg.train.seed)
    if args.max_files and len(files) > args.max_files:
        files = [files[i] for i in sorted(rng.choice(len(files), args.max_files, replace=False))]
    cols = list(dict.fromkeys(["file_id", "note_idx", "group_id", "pitch", "velocity", "baseline_velocity",
                               "velocity_delta", *model.feature_columns]))
    df = load_feature_table(data, cols, [args.split], [f["file_id"] for f in files])
    log.info("%s split: %d performances, %d notes", args.split, df["file_id"].nunique(), len(df))

    raw = model.predict_raw(df)
    base = df["baseline_velocity"].to_numpy(dtype=np.float64) if model.target_mode == "residual" else None
    pred = np.clip(model.to_velocity(raw, base), 1, 127)
    smoothed = None
    if cfg.smoothing.enabled:
        smoothed = np.empty_like(pred)
        for _, idx in df.groupby("file_id").indices.items():
            smoothed[idx] = apply_smoothing(pred[idx], df["group_id"].to_numpy()[idx], cfg.smoothing)
        smoothed = np.clip(smoothed, 1, 127)

    rows = compute_metrics(df, pred, model, smoothed)
    metric_cols = ["model", "mae", "rmse", "r2", "pearson", "spearman", "within_piece_r"]
    table = format_table(rows, metric_cols)
    print(f"\nMetrics on the {args.split} split (velocity units; oracle rows use the true velocities)")
    print(table)
    extra = ""
    if model.target_mode == "residual":
        d = regression_metrics(df["velocity_delta"].to_numpy(), raw)
        zero = regression_metrics(df["velocity_delta"].to_numpy(), np.zeros(len(raw)))
        extra = format_table([{"model": "model delta", **d}, {"model": "zero delta (= local baseline)", **zero}],
                             ["model", "mae", "rmse", "r2", "pearson"])
        print("\nResidual target (velocity - local baseline)")
        print(extra)

    per_file = (
        pd.DataFrame({"file_id": df["file_id"], "y": df["velocity"], "p": pred})
        .groupby("file_id")
        .apply(lambda g: pd.Series({"notes": len(g), "mae": np.mean(np.abs(g.p - g.y)),
                                    "pearson": np.corrcoef(g.y, g.p)[0, 1] if g.p.std() > 0 else np.nan,
                                    "true_mean": g.y.mean(), "pred_mean": g.p.mean()}), include_groups=False)
        .reset_index()
    )
    names = {f["file_id"]: (f["canonical_composer"], f["canonical_title"]) for f in fmeta["files"]}
    per_file["composer"] = per_file["file_id"].map(lambda i: names[i][0])
    per_file["title"] = per_file["file_id"].map(lambda i: names[i][1])
    per_file.sort_values("mae").to_csv(out / "per_file_metrics.csv", index=False)

    if not args.no_plots:
        make_plots(df, pred, model, out / "plots", args.split, args.shap_samples, cfg.train.seed)
        log.info("plots written to %s", out / "plots")

    listen_rows: list[dict[str, Any]] = []
    if args.n_midi > 0 and model.experiment != "performance_conditioned":
        maestro_dir = Path(args.maestro_dir or fmeta["maestro_dir"])
        chosen = [files[i] for i in sorted(rng.choice(len(files), min(args.n_midi, len(files)), replace=False))]
        listen_rows = musical_eval(model, chosen, maestro_dir, out / "midi", cfg.smoothing if args.smoothing else None)
        lcols = ["file", "notes", "beat_grid", "leakage_check", "flat64_mae", "humanized_mae", "humanized_r",
                 "true_sd", "humanized_sd"] + (["smoothed_mae", "smoothed_r"] if args.smoothing else [])
        print("\nListening-test files (humanized = predicted from the flat-64 input)")
        print(format_table(listen_rows, lcols, "{:.2f}"))
        pd.DataFrame(listen_rows).to_csv(out / "musical_eval.csv", index=False)

    save_json({"split": args.split, "metrics": rows, "musical_eval": listen_rows,
               "model": str(args.model), "target_mode": model.target_mode}, out / "metrics.json")
    md_parts = [f"# Evaluation: {args.model} on {args.split}\n", table]
    if extra:
        md_parts += ["\n## Residual target\n", extra]
    if listen_rows:
        md_parts += ["\n## Listening test\n", format_table(listen_rows, list(listen_rows[0]), "{:.2f}")]
    (out / "summary.md").write_text("\n".join(md_parts) + "\n", encoding="utf-8")
    log.info("results written to %s", out)


if __name__ == "__main__":
    main()
