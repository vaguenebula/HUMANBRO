"""Train the XGBoost velocity model (and optionally the A/B/C/D ablation).

    python train_xgboost.py --data data/features.parquet --output_dir models/velocity_xgb
    python train_xgboost.py --data data/features.parquet --output_dir models/velocity_xgb --ablation
    python train_xgboost.py --data data/features.parquet --output_dir models/velocity_xgb_residual --target_mode residual
    python train_xgboost.py --data data/features.parquet --output_dir models/velocity_xgb_pc \
        --experiment performance_conditioned          # analysis only: uses neighbour velocities

Splits come from the feature cache (official MAESTRO split, whole performances).
Early stopping watches the validation split; the test split is only reported.

Memory: each split is streamed from its Parquet parts into a single float32
matrix holding only the columns the current model needs (~3 GB for the full
training split), and every feature set of the ablation is loaded on its own.
"""

from __future__ import annotations

import argparse
import gc
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from config import Config, add_config_args, config_from_dict, load_config
from midi_features import (
    FEATURE_SET_DESCRIPTIONS,
    PC_COLUMNS,
    assert_no_leakage,
    select_feature_columns,
)
from model_io import meta_path_for
from utils import (
    SPLITS,
    choose_files,
    format_table,
    load_features_meta,
    load_split_matrix,
    regression_metrics,
    resolve_device,
    save_json,
    set_seed,
    setup_logging,
    split_stats_table,
    within_piece_pearson,
)

log = logging.getLogger("train_xgboost")

INFO_COLUMNS = ["file_id", "pitch", "velocity", "baseline_velocity", "velocity_delta"]


def train_stats(info: pd.DataFrame) -> dict[str, Any]:
    """Statistics of the TRAIN split only, used for the simple baselines."""
    v = info["velocity"].to_numpy(dtype=np.float64)
    pitch = info["pitch"].to_numpy().astype(np.int64)
    sums = np.bincount(pitch, weights=v, minlength=128)
    counts = np.bincount(pitch, minlength=128)
    per_pitch = np.where(counts > 0, sums / np.maximum(counts, 1), v.mean())
    return {
        "global_mean": float(v.mean()),
        "global_median": float(np.median(v)),
        "per_pitch_mean": per_pitch.tolist(),
        "per_pitch_count": counts.tolist(),
    }


def evaluate_split(pred_raw: np.ndarray, info: pd.DataFrame, target_mode: str) -> dict[str, Any]:
    """Metrics in velocity units. Residual models are reconstructed with the (oracle) local baseline."""
    vel = info["velocity"].to_numpy(dtype=np.float64)
    out: dict[str, Any] = {}
    if target_mode == "absolute":
        pred_vel = pred_raw
    else:
        pred_vel = info["baseline_velocity"].to_numpy(dtype=np.float64) + pred_raw
        out["delta"] = regression_metrics(info["velocity_delta"].to_numpy(), pred_raw)
    pred_vel = np.clip(pred_vel, 1, 127)
    out["velocity"] = regression_metrics(vel, pred_vel)
    out["velocity"]["within_piece_pearson"] = within_piece_pearson(vel, pred_vel, info["file_id"].to_numpy())
    return out


def importance_table(booster: xgb.Booster, columns: list[str]) -> pd.DataFrame:
    kinds = ["gain", "total_gain", "weight", "cover"]
    scores = {k: booster.get_score(importance_type=k) for k in kinds}
    df = pd.DataFrame({k: [scores[k].get(c, 0.0) for c in columns] for k in kinds}, index=columns)
    df.index.name = "feature"
    return df.sort_values("gain", ascending=False)


def target_of(info: pd.DataFrame, target_mode: str) -> np.ndarray:
    col = "velocity" if target_mode == "absolute" else "velocity_delta"
    return info[col].to_numpy(dtype=np.float32)


def train_one(
    data: Path,
    columns: list[str],
    feature_set: str,
    target_mode: str,
    cfg: Config,
    device: str,
    out_dir: Path,
    file_ids: list[int] | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    t_load = time.time()
    x_tr, info_tr = load_split_matrix(data, "train", columns, INFO_COLUMNS, file_ids)
    x_va, info_va = load_split_matrix(data, "validation", columns, INFO_COLUMNS, file_ids)
    log.info("[%s] loaded train %s + validation %s float32 matrices in %.1f s (%.2f GB)", feature_set, x_tr.shape,
             x_va.shape, time.time() - t_load, (x_tr.nbytes + x_va.nbytes) / 1e9)
    xc = cfg.xgb
    params = dict(
        objective=xc.objective,
        n_estimators=xc.n_estimators,
        learning_rate=xc.learning_rate,
        max_depth=xc.max_depth,
        subsample=xc.subsample,
        colsample_bytree=xc.colsample_bytree,
        min_child_weight=xc.min_child_weight,
        reg_alpha=xc.reg_alpha,
        reg_lambda=xc.reg_lambda,
        gamma=xc.gamma,
        max_bin=xc.max_bin,
        tree_method="hist",
        device=device,
        n_jobs=xc.n_jobs,
        random_state=cfg.train.seed,
        early_stopping_rounds=xc.early_stopping_rounds,
        eval_metric=["mae", "rmse"],  # early stopping uses the last one (rmse, matches the objective)
    )
    log.info("[%s] training on %d notes x %d features (%s), target=%s, device=%s", feature_set, len(x_tr),
             len(columns), FEATURE_SET_DESCRIPTIONS.get(feature_set, ""), target_mode, device)
    model = xgb.XGBRegressor(**params)
    t0 = time.time()
    model.fit(x_tr, target_of(info_tr, target_mode), eval_set=[(x_va, target_of(info_va, target_mode))],
              verbose=cfg.train.verbose_every)
    seconds = time.time() - t0
    del x_tr, info_tr
    gc.collect()

    best_it = int(model.best_iteration)
    booster = model.get_booster()[: best_it + 1]  # keep only the trees up to the best iteration
    booster.feature_names = columns
    booster.set_param({"device": "cpu"})
    del model

    metrics: dict[str, Any] = {"validation": evaluate_split(booster.inplace_predict(x_va), info_va, target_mode)}
    del x_va
    x_te, info_te = load_split_matrix(data, "test", columns, INFO_COLUMNS, file_ids)
    if len(x_te):
        metrics["test"] = evaluate_split(booster.inplace_predict(x_te), info_te, target_mode)
    del x_te

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.json"
    booster.save_model(str(model_path))
    imp = importance_table(booster, columns)
    imp.to_csv(out_dir / "feature_importance.csv")
    meta = {
        "model_file": model_path.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_set": feature_set,
        "feature_set_description": FEATURE_SET_DESCRIPTIONS.get(feature_set, ""),
        "experiment": cfg.train.experiment,
        "target_mode": target_mode,
        "feature_columns": columns,
        "best_iteration": best_it,
        "n_trees": best_it + 1,
        "train_seconds": seconds,
        "device": device,
        "xgb_params": {k: v for k, v in params.items() if k != "eval_metric"},
        "metrics": metrics,
        **context,
    }
    save_json(meta, meta_path_for(model_path))
    save_json(metrics, out_dir / "metrics.json")
    log.info("[%s] best iteration %d, %.0f s; saved %s", feature_set, best_it, seconds, model_path)
    for split, m in metrics.items():
        v = m["velocity"]
        log.info("[%s] %-10s MAE %.2f  RMSE %.2f  R2 %.3f  r %.3f  within-piece r %.3f", feature_set, split,
                 v["mae"], v["rmse"], v["r2"], v["pearson"], v["within_piece_pearson"])
    gc.collect()
    return {"feature_set": feature_set, "metrics": metrics, "importance": imp, "model_path": model_path,
            "n_features": len(columns)}


def baseline_rows(infos: dict[str, pd.DataFrame], stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    per_pitch = np.asarray(stats["per_pitch_mean"])
    for name, fn in (
        ("baseline: global mean", lambda p: np.full(len(p), stats["global_mean"])),
        ("baseline: per-pitch mean", lambda p: per_pitch[p["pitch"].to_numpy().astype(int)]),
    ):
        row: dict[str, Any] = {"model": name, "features": 0}
        for split in ("validation", "test"):
            part = infos[split]
            if len(part):
                m = regression_metrics(part["velocity"].to_numpy(), fn(part))
                row[f"{split}_mae"], row[f"{split}_r"] = m["mae"], m["pearson"]
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="feature cache directory from build_dataset.py")
    ap.add_argument("--output_dir", default="models/velocity_xgb")
    ap.add_argument("--target_mode", choices=["absolute", "residual"], default=None)
    ap.add_argument("--feature_set", choices=["A", "B", "C", "D"], default=None)
    ap.add_argument("--experiment", choices=["primary", "performance_conditioned"], default=None)
    ap.add_argument("--ablation", action="store_true", help="also train feature sets A-D and print a comparison")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    ap.add_argument("--max_files_per_split", type=int, default=None, help="subsample whole files for quick runs")
    add_config_args(ap)
    args = ap.parse_args()
    setup_logging()

    cfg = load_config(args.config, args.set)
    overrides: dict[str, dict[str, Any]] = {"train": {}, "xgb": {}, "target": {}}
    if args.feature_set:
        overrides["train"]["feature_set"] = args.feature_set
    if args.experiment:
        overrides["train"]["experiment"] = args.experiment
    if args.max_files_per_split is not None:
        overrides["train"]["max_files_per_split"] = args.max_files_per_split
    if args.device:
        overrides["xgb"]["device"] = args.device
    if args.target_mode:
        overrides["target"]["mode"] = args.target_mode
    cfg = config_from_dict(overrides, base=cfg)
    set_seed(cfg.train.seed)

    data = Path(args.data or cfg.data.features_path)
    out_dir = Path(args.output_dir)
    fmeta = load_features_meta(data)
    groups: dict[str, str] = fmeta["feature_groups"]
    target_mode = cfg.target.mode
    experiment = cfg.train.experiment
    primary = cfg.train.feature_set
    sets = [primary] + ([s for s in "ABCD" if s != primary] if args.ablation else [])

    columns_by_set: dict[str, list[str]] = {}
    for s in sets:
        cols = select_feature_columns(groups, s)
        if experiment == "performance_conditioned":
            cols = cols + PC_COLUMNS
        assert_no_leakage(cols, experiment)
        columns_by_set[s] = cols
    if experiment == "performance_conditioned":
        log.warning("PERFORMANCE-CONDITIONED experiment: inputs include neighbour/chord-mate velocities. "
                    "Not a score-to-performance model; cannot humanize deadpan MIDI.")

    file_ids = None
    if cfg.train.max_files_per_split is not None:
        file_ids = choose_files(fmeta["files"], SPLITS, cfg.train.max_files_per_split, cfg.train.seed)

    # Split bookkeeping from the light columns only.
    infos = {s: load_split_matrix(data, s, [], INFO_COLUMNS, file_ids)[1] for s in SPLITS}
    files_per_split = {s: set(infos[s]["file_id"].unique()) for s in SPLITS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not (files_per_split[a] & files_per_split[b]), f"performance overlap between {a} and {b}"
    print("\nSplit summary (whole performances per split; velocity distribution)")
    print(split_stats_table(pd.concat(infos.values(), ignore_index=True)))
    print()

    stats = train_stats(infos["train"])
    device = resolve_device(cfg.xgb.device)
    context = {
        "pipeline_config": fmeta["pipeline_config"],
        "features_data": str(data.resolve()),
        "maestro_dir": fmeta.get("maestro_dir"),
        "train_stats": stats,
        "n_files": {s: len(files_per_split[s]) for s in SPLITS},
        "n_rows": {s: len(infos[s]) for s in SPLITS},
        "config": cfg.to_dict(),
        "versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
    }

    results = []
    for s in sets:
        dest = out_dir if s == primary else out_dir / "ablation" / s
        results.append(train_one(data, columns_by_set[s], s, target_mode, cfg, device, dest, file_ids, context))

    top = results[0]["importance"].head(30)
    print(f"\nTop 30 features by gain (feature set {primary}):")
    print(format_table([{"feature": f, "gain": r.gain, "weight": r.weight} for f, r in top.iterrows()],
                       ["feature", "gain", "weight"], "{:.1f}"))

    if args.ablation:
        rows = baseline_rows(infos, stats)
        for r in sorted(results, key=lambda r: r["feature_set"]):
            row: dict[str, Any] = {"model": f"{r['feature_set']}: {FEATURE_SET_DESCRIPTIONS[r['feature_set']]}",
                                   "features": r["n_features"]}
            for split in ("validation", "test"):
                if split in r["metrics"]:
                    v = r["metrics"][split]["velocity"]
                    row[f"{split}_mae"], row[f"{split}_r"] = v["mae"], v["pearson"]
                    row[f"{split}_piece_r"] = v["within_piece_pearson"]
            rows.append(row)
        cols = ["model", "features", "validation_mae", "validation_r", "validation_piece_r", "test_mae", "test_r",
                "test_piece_r"]
        table = format_table(rows, cols)
        print(f"\nAblation (target={target_mode}, experiment={experiment}; *_piece_r = mean within-performance r)")
        print(table)
        pd.DataFrame(rows).to_csv(out_dir / "ablation.csv", index=False)
        (out_dir / "ablation.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
