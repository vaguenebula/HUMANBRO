"""Train the micro-timing model: multi-quantile XGBoost on onset offsets (ms).

    python build_dataset.py --task timing --maestro_dir data/maestro-v3.0.0 --output data/timing.parquet
    python train_timing.py --data data/timing.parquet --output_dir models/timing_xgb
    python train_timing.py ... --ablation --max_files_per_split 200      # feature sets A-D, quick
    python train_timing.py --refit_sampler models/timing_xgb/model.json  # refit only the sampler

One booster predicts every quantile in ``timing_model.quantiles`` (XGBoost
``reg:quantileerror``). Early stopping watches the mean pinball loss on the validation
split; the test split is only reported. Afterwards the sampling copula
(timing_sampling.py) is fitted on the validation split's normal scores.

Outputs in --output_dir: model.json, model.meta.json (quantiles, copula, metrics, pipeline
config), feature_importance.csv, metrics.json.
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
from midi_features import FEATURE_SET_DESCRIPTIONS, select_feature_columns
from model_io import meta_path_for
from timing_features import assert_no_timing_leakage
from timing_metrics import distribution_metrics, median_metrics
from timing_sampling import fit_sampler, normal_knots, normal_scores, sort_quantiles
from train_xgboost import importance_table
from utils import (
    SPLITS,
    choose_files,
    format_table,
    load_features_meta,
    load_split_matrix,
    resolve_device,
    save_json,
    set_seed,
    setup_logging,
)

log = logging.getLogger("train_timing")

INFO_COLUMNS = ["file_id", "group_id", "offset_ms", "onset_beat", "is_chord_top"]


def split_metrics(q: np.ndarray, info: pd.DataFrame, alphas: np.ndarray, uncond: np.ndarray, mid: int) -> dict[str, Any]:
    y = info["offset_ms"].to_numpy(dtype=np.float64)
    f, g, top = info["file_id"].to_numpy(), info["group_id"].to_numpy(), info["is_chord_top"].to_numpy()
    return {
        "notes": int(len(y)),
        "model": distribution_metrics(y, q, alphas),
        "unconditional": distribution_metrics(y, uncond, alphas),
        "median": median_metrics(y, q[:, mid], f, g, top),
        "zero_offset": median_metrics(y, np.zeros(len(y)), f, g, top)["all"],
    }


def log_split(tag: str, split: str, m: dict[str, Any]) -> None:
    md = m["median"]
    log.info("[%s] %-10s pinball %.2f (unconditional %.2f) | median MAE %.2f ms (zero %.2f), r %.3f, "
             "event r %.3f, asynchrony r %.3f, top lead %.1f ms (true %.1f)", tag, split, m["model"]["mean_pinball"],
             m["unconditional"]["mean_pinball"], md["all"]["mae"], m["zero_offset"]["mae"], md["all"]["pearson"],
             md["event"]["pearson"], md["asynchrony"]["pearson"], md["top_lead_ms"]["pred"], md["top_lead_ms"]["true"])


def fit_validation_extras(q_va: np.ndarray, info_va: pd.DataFrame, alphas: np.ndarray, seed: int) -> dict[str, Any]:
    """Everything fitted on validation predictions: the sampler, and the residual quantiles of a
    homoscedastic baseline (model median + one fixed residual distribution) for evaluation."""
    y = info_va["offset_ms"].to_numpy(dtype=np.float64)
    z = normal_scores(q_va, normal_knots(alphas), y)
    copula = fit_sampler(z, info_va["file_id"].to_numpy(), info_va["group_id"].to_numpy(),
                         info_va["onset_beat"].to_numpy(), seed=seed)
    mid = int(np.argmin(np.abs(alphas - 0.5)))
    resid = np.quantile(y - q_va[:, mid], alphas)
    log.info("sampler from validation normal scores: within-chord rho %.3f, correlation length %.2f beats; "
             "raw score sd %.3f (1 = calibrated)", copula.rho, copula.ell, float(np.std(z)))
    return {"copula": copula.to_dict(), "median_residual_quantiles_ms": resid.tolist()}


def train_one(data: Path, columns: list[str], feature_set: str, cfg: Config, device: str, out_dir: Path,
              file_ids: list[int] | None, context: dict[str, Any]) -> dict[str, Any]:
    tm = cfg.timing_model
    alphas = np.asarray(tm.quantiles, dtype=np.float64)
    mid = int(np.argmin(np.abs(alphas - 0.5)))
    t_load = time.time()
    x_tr, info_tr = load_split_matrix(data, "train", columns, INFO_COLUMNS, file_ids)
    x_va, info_va = load_split_matrix(data, "validation", columns, INFO_COLUMNS, file_ids)
    log.info("[%s] loaded train %s + validation %s in %.1f s (%.2f GB)", feature_set, x_tr.shape, x_va.shape,
             time.time() - t_load, (x_tr.nbytes + x_va.nbytes) / 1e9)
    y_tr = info_tr["offset_ms"].to_numpy(dtype=np.float32)
    y_va = info_va["offset_ms"].to_numpy(dtype=np.float32)
    # Offsets this far from the reference beat are nearly always beat-tracking errors.
    w_tr = (np.abs(y_tr) <= tm.max_abs_offset_ms).astype(np.float32)
    w_va = (np.abs(y_va) <= tm.max_abs_offset_ms).astype(np.float32)
    uncond = np.quantile(y_tr[w_tr > 0], alphas)

    params = dict(
        objective="reg:quantileerror",
        quantile_alpha=alphas,
        n_estimators=tm.n_estimators,
        learning_rate=tm.learning_rate,
        max_depth=tm.max_depth,
        subsample=tm.subsample,
        colsample_bytree=tm.colsample_bytree,
        min_child_weight=tm.min_child_weight,
        reg_lambda=tm.reg_lambda,
        max_bin=tm.max_bin,
        tree_method="hist",
        device=device,
        n_jobs=cfg.xgb.n_jobs,
        random_state=cfg.train.seed,
        early_stopping_rounds=tm.early_stopping_rounds,
        eval_metric="quantile",  # mean pinball loss over the quantile levels
    )
    log.info("[%s] training on %d notes (%d down-weighted to 0: |offset| > %g ms) x %d features (%s), "
             "%d quantiles, device=%s", feature_set, len(y_tr), int((w_tr == 0).sum()), tm.max_abs_offset_ms,
             len(columns), FEATURE_SET_DESCRIPTIONS.get(feature_set, ""), len(alphas), device)
    model = xgb.XGBRegressor(**params)
    t0 = time.time()
    model.fit(x_tr, y_tr, sample_weight=w_tr, eval_set=[(x_va, y_va)], sample_weight_eval_set=[w_va],
              verbose=cfg.train.verbose_every)
    seconds = time.time() - t0
    del x_tr, info_tr, y_tr, w_tr
    gc.collect()

    best_it = int(model.best_iteration)
    booster = model.get_booster()[: best_it + 1]  # keep the rounds up to the best iteration
    booster.feature_names = columns
    booster.set_param({"device": "cpu"})
    del model

    q_va = sort_quantiles(booster.inplace_predict(x_va))
    del x_va
    extras = fit_validation_extras(q_va, info_va, alphas, cfg.train.seed)

    metrics: dict[str, Any] = {"validation": split_metrics(q_va, info_va, alphas, uncond, mid)}
    x_te, info_te = load_split_matrix(data, "test", columns, INFO_COLUMNS, file_ids)
    if len(x_te):
        metrics["test"] = split_metrics(sort_quantiles(booster.inplace_predict(x_te)), info_te, alphas, uncond, mid)
    del x_te

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.json"
    booster.save_model(str(model_path))
    imp = importance_table(booster, columns)
    imp.to_csv(out_dir / "feature_importance.csv")
    meta = {
        "task": "timing",
        "model_file": model_path.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "offset_ms: performed onset minus the reference beat time of its score position",
        "feature_set": feature_set,
        "feature_set_description": FEATURE_SET_DESCRIPTIONS.get(feature_set, ""),
        "feature_columns": columns,
        "quantiles": alphas.tolist(),
        "unconditional_quantiles_ms": uncond.tolist(),
        **extras,
        "best_iteration": best_it,
        "n_rounds": best_it + 1,
        "train_seconds": seconds,
        "device": device,
        "xgb_params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()},
        "metrics": metrics,
        **context,
    }
    save_json(meta, meta_path_for(model_path))
    save_json(metrics, out_dir / "metrics.json")
    log.info("[%s] best iteration %d, %.0f s; saved %s", feature_set, best_it, seconds, model_path)
    for split, m in metrics.items():
        log_split(feature_set, split, m)
    gc.collect()
    return {"feature_set": feature_set, "metrics": metrics, "importance": imp, "n_features": len(columns)}


def refit_sampler(model_path: Path, data: Path | None, seed: int) -> None:
    """Refit the sampling copula + recalibration of a trained model on its validation split."""
    from model_io import TimingModel

    model = TimingModel.load(model_path)
    data = data or Path(model.meta["features_data"])
    x_va, info_va = load_split_matrix(data, "validation", model.feature_columns, INFO_COLUMNS)
    q_va = sort_quantiles(model.booster.inplace_predict(x_va))
    save_json({**model.meta, **fit_validation_extras(q_va, info_va, model.quantiles, seed)}, meta_path_for(model_path))
    log.info("refit sampler for %s", model_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/timing.parquet", help="timing feature cache (build_dataset.py --task timing)")
    ap.add_argument("--output_dir", default="models/timing_xgb")
    ap.add_argument("--feature_set", choices=["A", "B", "C", "D"], default=None)
    ap.add_argument("--ablation", action="store_true", help="also train feature sets A-D and print a comparison")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    ap.add_argument("--max_files_per_split", type=int, default=None, help="subsample whole files for quick runs")
    ap.add_argument("--refit_sampler", default=None, metavar="MODEL_JSON",
                    help="only refit the sampling copula/recalibration of this trained model, then exit")
    add_config_args(ap)
    args = ap.parse_args()
    setup_logging()
    if args.refit_sampler:
        refit_sampler(Path(args.refit_sampler), None, load_config(args.config, args.set).train.seed)
        return

    cfg = load_config(args.config, args.set)
    overrides: dict[str, dict[str, Any]] = {"train": {}, "xgb": {}, "timing_model": {}}
    if args.feature_set:
        overrides["timing_model"]["feature_set"] = args.feature_set
    if args.max_files_per_split is not None:
        overrides["train"]["max_files_per_split"] = args.max_files_per_split
    if args.device:
        overrides["xgb"]["device"] = args.device
    cfg = config_from_dict(overrides, base=cfg)
    set_seed(cfg.train.seed)

    data = Path(args.data)
    out_dir = Path(args.output_dir)
    fmeta = load_features_meta(data)
    if fmeta.get("task") != "timing":
        raise SystemExit(f"{data} is not a timing dataset; build one with build_dataset.py --task timing")
    groups: dict[str, str] = fmeta["feature_groups"]
    primary = cfg.timing_model.feature_set
    sets = [primary] + ([s for s in "ABCD" if s != primary] if args.ablation else [])
    columns_by_set = {}
    for s in sets:
        cols = select_feature_columns(groups, s)
        assert_no_timing_leakage(cols)
        columns_by_set[s] = cols

    file_ids = None
    if cfg.train.max_files_per_split is not None:
        file_ids = choose_files(fmeta["files"], SPLITS, cfg.train.max_files_per_split, cfg.train.seed)
    infos = {s: load_split_matrix(data, s, [], ["file_id"], file_ids)[1] for s in SPLITS}
    files_per_split = {s: set(infos[s]["file_id"].unique()) for s in SPLITS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not (files_per_split[a] & files_per_split[b]), f"performance overlap between {a} and {b}"

    device = resolve_device(cfg.xgb.device)
    context = {
        "pipeline_config": fmeta["pipeline_config"],
        "features_data": str(data.resolve()),
        "maestro_dir": fmeta.get("maestro_dir"),
        "n_files": {s: len(files_per_split[s]) for s in SPLITS},
        "n_rows": {s: len(infos[s]) for s in SPLITS},
        "config": cfg.to_dict(),
        "versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    del infos

    results = []
    for s in sets:
        dest = out_dir if s == primary else out_dir / "ablation" / s
        results.append(train_one(data, columns_by_set[s], s, cfg, device, dest, file_ids, context))

    top = results[0]["importance"].head(25)
    print(f"\nTop 25 features by gain (feature set {primary}):")
    print(format_table([{"feature": f, "gain": r.gain, "weight": r.weight} for f, r in top.iterrows()],
                       ["feature", "gain", "weight"], "{:.1f}"))

    rows = []
    for r in sorted(results, key=lambda r: r["feature_set"]):
        for split in ("validation", "test"):
            if split not in r["metrics"]:
                continue
            m = r["metrics"][split]
            rows.append({
                "model": f"{r['feature_set']}: {FEATURE_SET_DESCRIPTIONS[r['feature_set']]}", "split": split,
                "features": r["n_features"], "pinball": m["model"]["mean_pinball"],
                "uncond_pinball": m["unconditional"]["mean_pinball"], "median_mae": m["median"]["all"]["mae"],
                "zero_mae": m["zero_offset"]["mae"], "median_r": m["median"]["all"]["pearson"],
                "event_r": m["median"]["event"]["pearson"], "async_r": m["median"]["asynchrony"]["pearson"],
            })
    table = format_table(rows, ["model", "split", "features", "pinball", "uncond_pinball", "median_mae", "zero_mae",
                                "median_r", "event_r", "async_r"])
    print("\nTiming model (offsets in ms; pinball = mean over quantile levels, lower is better)")
    print(table)
    if args.ablation:
        pd.DataFrame(rows).to_csv(out_dir / "ablation.csv", index=False)
        (out_dir / "ablation.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
