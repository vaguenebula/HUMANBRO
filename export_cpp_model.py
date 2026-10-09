"""Export a trained model to the compact binary format read by the C++ runtime (cpp/).

    python export_cpp_model.py --model models/velocity_xgb_quantized/model.json
    python export_cpp_model.py --model models/timing_xgb/model.json            # timing (quantile) model
    python export_cpp_model.py --model models/velocity_xgb/model.json --output my_model.hbm

Writes ``<model>.hbm`` (default: next to model.json). File layout, little-endian:

    char[4] "HBRO", u32 version (2)
    u32 header length, header bytes: UTF-8 "key=value" lines holding the task (velocity or
        timing), the feature-pipeline settings the model was trained with, base_score, the
        number of outputs and the feature names in model column order. Timing models add
        their quantile levels, normal-score knots, sampler (copula + recalibration) and the
        settings of the optional grid snap.
    u32 number of trees, then per tree:
        u32 output; u32 n_nodes; i32 left[n]; i32 right[n]; i32 feature[n]; f32 value[n];
        u8 default_left[n]
        (value = split threshold for internal nodes, leaf value where left == -1)

Version 1 files (velocity models exported before) have no ``output`` field; the C++ loader
reads both. Prediction for output k = base_score + the leaves of the trees of output k,
accumulated in float32 in tree order, which is what XGBoost computes for
reg:squarederror and reg:quantileerror. The export is verified against XGBoost on rows
from the model's own feature cache before the file is kept.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
from pathlib import Path

import numpy as np

from model_io import TimingModel, VelocityModel, meta_path_for
from timing_sampling import normal_knots
from utils import load_feature_table, load_features_meta, load_json, setup_logging

log = logging.getLogger("export_cpp_model")

MAGIC = b"HBRO"
VERSION = 2
OBJECTIVES = {"reg:squarederror": "velocity", "reg:quantileerror": "timing"}


def read_trees(model_path: Path) -> tuple[float, int, list[dict[str, np.ndarray]]]:
    """(base_score, number of outputs, trees); each tree dict has its ``output`` index."""
    with open(model_path, encoding="utf-8") as fh:
        learner = json.load(fh)["learner"]
    objective = learner["objective"]["name"]
    if objective not in OBJECTIVES:
        raise SystemExit(f"unsupported objective {objective} (supported: {sorted(OBJECTIVES)})")
    params = learner["learner_model_param"]
    base_score = float(str(params["base_score"]).strip("[]"))
    num_outputs = max(1, int(params.get("num_target", 1)))
    booster = learner["gradient_booster"]
    if booster["name"] != "gbtree":
        raise SystemExit(f"only gbtree boosters are supported (got {booster['name']})")
    if int(booster["model"]["gbtree_model_param"]["num_parallel_tree"]) != 1:
        raise SystemExit("random-forest style boosters (num_parallel_tree > 1) are not supported")
    tree_info = booster["model"]["tree_info"]
    trees = []
    for t, out in zip(booster["model"]["trees"], tree_info):
        if any(int(s) != 0 for s in t["split_type"]):
            raise SystemExit("categorical splits are not supported")
        if int(t["tree_param"]["size_leaf_vector"]) > 1:
            raise SystemExit("vector-leaf (multi_output_tree) models are not supported")
        trees.append(
            {
                "output": int(out),
                "left": np.asarray(t["left_children"], dtype="<i4"),
                "right": np.asarray(t["right_children"], dtype="<i4"),
                "feature": np.asarray(t["split_indices"], dtype="<i4"),
                "value": np.asarray(t["split_conditions"], dtype="<f4"),
                "default_left": np.asarray(t["default_left"], dtype=np.uint8),
            }
        )
    return base_score, num_outputs, trees


def predict_numpy(base_score: float, num_outputs: int, trees: list[dict[str, np.ndarray]], x: np.ndarray) -> np.ndarray:
    """Reference evaluator with exactly the semantics the C++ runtime implements. Shape (rows, outputs)."""
    x = np.asarray(x, dtype=np.float32)
    rows = np.arange(len(x))
    acc = np.full((len(x), num_outputs), np.float32(base_score), dtype=np.float32)
    for t in trees:
        node = np.zeros(len(x), dtype=np.int64)
        while True:
            internal = t["left"][node] != -1
            if not internal.any():
                break
            r, nd = rows[internal], node[internal]
            v = x[r, t["feature"][nd]]
            go_left = np.where(np.isnan(v), t["default_left"][nd] == 1, v < t["value"][nd])
            node[internal] = np.where(go_left, t["left"][nd], t["right"][nd])
        k = t["output"]
        acc[:, k] = (acc[:, k] + t["value"][node]).astype(np.float32)
    return acc


def _floats(xs) -> str:
    return " ".join(repr(float(v)) for v in xs)


def feature_header(pc: dict) -> dict[str, str]:
    f = pc["features"]
    return {
        "context": f["context"],
        "chord_tolerance_s": repr(float(f["chord_tolerance_s"])),
        "time_windows_s": _floats(f["time_windows_s"]),
        "beat_windows": _floats(f["beat_windows"]),
        "nearest_n": " ".join(str(int(v)) for v in f["nearest_n"]),
        "strong_beat_tolerance": repr(float(f["strong_beat_tolerance"])),
        "rest_min_s": repr(float(f["rest_min_s"])),
        "phrase_gap_beats": repr(float(f["phrase_gap_beats"])),
        "phrase_gap_min_s": repr(float(f["phrase_gap_min_s"])),
        "include_piece_features": str(int(bool(f["include_piece_features"]))),
    }


def header_text(model: VelocityModel | TimingModel, task: str, base_score: float, num_outputs: int) -> str:
    pc = model.meta["pipeline_config"]
    lines = {"task": task, **feature_header(pc)}
    if task == "velocity":
        f, tg = pc["features"], pc["target"]
        lines.update({
            "target_mode": model.target_mode,
            "experiment": model.experiment,
            "quantize": str(int(bool(f["quantize"]))),
            "quantize_subdivisions": " ".join(str(int(v)) for v in f["quantize_subdivisions"]),
            "residual_window_notes": str(int(tg["residual_window_notes"])),
            "residual_stat": tg["residual_stat"],
        })
    else:
        tc = pc["timing"]
        cop = model.copula
        if not cop.calib_u:
            raise SystemExit("this timing model has no sampler recalibration; run "
                             "train_timing.py --refit_sampler on it first")
        lines.update({
            "target_mode": "offset_ms",
            "experiment": "primary",
            "quantize": "0",
            "quantize_subdivisions": "",
            "timing_subdivisions": " ".join(str(int(v)) for v in tc["quantize_subdivisions"]),
            "timing_switch_cost": repr(float(tc["subdivision_switch_cost"])),
            "quantile_levels": _floats(model.quantiles),
            "quantile_knots": _floats(normal_knots(model.quantiles)),
            "copula_rho": repr(float(cop.rho)),
            "copula_ell": repr(float(cop.ell)),
            "calib_u": _floats(cop.calib_u),
            "calib_z": _floats(cop.calib_z),
        })
    lines.update({
        "num_outputs": str(num_outputs),
        "base_score": repr(float(np.float32(base_score))),
        "features": " ".join(model.feature_columns),
    })
    return "".join(f"{k}={v}\n" for k, v in lines.items())


def write_hbm(path: Path, header: str, trees: list[dict[str, np.ndarray]]) -> None:
    with open(path, "wb") as fh:
        hb = header.encode("utf-8")
        fh.write(MAGIC + struct.pack("<II", VERSION, len(hb)) + hb)
        fh.write(struct.pack("<I", len(trees)))
        for t in trees:
            fh.write(struct.pack("<II", t["output"], len(t["left"])))
            for key in ("left", "right", "feature", "value", "default_left"):
                fh.write(t[key].tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model.json from train_xgboost.py or train_timing.py")
    ap.add_argument("--output", default=None, help="default: <model>.hbm next to model.json")
    ap.add_argument("--check_rows", type=int, default=20000, help="rows used to verify the export (0 = skip)")
    args = ap.parse_args()
    setup_logging()

    model_path = Path(args.model)
    base_score, num_outputs, trees = read_trees(model_path)
    task = "timing" if load_json(meta_path_for(model_path)).get("task") == "timing" else "velocity"
    model = TimingModel.load(model_path) if task == "timing" else VelocityModel.load(model_path)
    if task == "velocity" and model.experiment == "performance_conditioned":
        raise SystemExit("performance-conditioned models need true velocities as input; not exportable for humanizing")
    if model.meta["pipeline_config"]["features"]["context"] != "bidirectional":
        raise SystemExit("the C++ runtime implements the bidirectional feature set only")
    if task == "timing" and num_outputs != len(model.quantiles):
        raise SystemExit(f"{num_outputs} outputs but {len(model.quantiles)} quantile levels")
    log.info("%s model: %d trees, %d outputs, %d nodes, base_score %.6g, %d features", task, len(trees), num_outputs,
             sum(len(t["left"]) for t in trees), base_score, len(model.feature_columns))

    if args.check_rows > 0:
        data = Path(model.meta["features_data"])
        fmeta = load_features_meta(data)
        ids = [f["file_id"] for f in fmeta["files"] if f["split"] == "test"][:40]
        df = load_feature_table(data, model.feature_columns, ["test"], ids).head(args.check_rows)
        x = df[model.feature_columns].to_numpy(np.float32)
        ref = np.asarray(model.booster.inplace_predict(x), dtype=np.float64).reshape(len(df), -1)
        mine = predict_numpy(base_score, num_outputs, trees, x)
        diff = float(np.max(np.abs(ref - mine)))
        log.info("export check on %d rows: max |xgboost - exported| = %.3g", len(df), diff)
        if diff > 1e-3:
            raise SystemExit("exported trees do not reproduce XGBoost; not writing the file")

    out = Path(args.output) if args.output else model_path.with_suffix(".hbm")
    write_hbm(out, header_text(model, task, base_score, num_outputs), trees)
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
