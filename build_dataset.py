"""Extract per-note features from every MAESTRO performance and cache them as Parquet.

    python build_dataset.py --maestro_dir data/maestro-v3.0.0 --output data/features.parquet
    python build_dataset.py --maestro_dir maestro-v3.0.0-midi.zip --output data/features.parquet
    python build_dataset.py ... --quantize --output data/features_quantized.parquet   # score-like experiment
    python build_dataset.py ... --task timing --output data/timing.parquet           # micro-timing targets

Output: a hive-partitioned Parquet dataset (a directory)

    data/features.parquet/
        _meta.json                          feature list + groups, config, per-file info, failures, stats
        split=train/part-00012.parquet      one file per performance, rows in canonical note order
        split=validation/...
        split=test/...

* The official MAESTRO train/validation/test split is used, so every note of a
  performance (and every performance of a composition) stays in one split.
* One Parquet part per performance makes the build resumable: re-running skips
  parts that already exist (unless --overwrite) as long as the config matches.
* Malformed MIDI files are logged in _meta.json["failures"] and skipped.
* ``--task timing`` stores features of the reconstructed score and the onset offsets
  (``offset_ms``) instead of velocity targets; see timing_features.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from config import Config, add_config_args, config_from_dict, load_config
from download_or_locate_maestro import locate_maestro, read_metadata
from midi_features import (
    META_COLUMNS,
    PC_COLUMNS,
    TARGET_COLUMNS,
    canonical_velocities,
    compute_targets,
    featurize_midi,
    performance_conditioned_features,
)
from midi_io import load_midi
from timing_features import TIMING_META_COLUMNS, TIMING_TARGET_COLUMNS, featurize_performance
from utils import (
    SPLITS,
    format_table,
    load_json,
    save_json,
    set_seed,
    setup_logging,
    stats_from_histogram,
)

log = logging.getLogger("build_dataset")

_INFO_KEY = b"humanbro_file_info"
_CFG: Config | None = None
_TASK = "velocity"
OFFSET_HIST_MS = 300  # timing summary histogram: 1 ms bins over +/- this range (clipped)


def _init_worker(cfg_dict: dict[str, Any], task: str) -> None:
    global _CFG, _TASK
    _CFG = config_from_dict(cfg_dict)
    _TASK = task
    setup_logging("WARNING")


def _velocity_rows(md: Any, cfg: Config) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    pf = featurize_midi(md, cfg)
    group_id = pf.df["group_id"].to_numpy()
    vel = canonical_velocities(md, pf.df)  # target only; features above never saw it
    targets = compute_targets(vel, group_id, cfg.target)
    pc = performance_conditioned_features(vel, group_id, targets["baseline_velocity"].to_numpy())
    hist = np.bincount(vel.astype(np.int64), minlength=128).tolist()
    return pd.concat([pf.df, targets, pc], axis=1), pf, {"target_hist": hist, "feature_groups": pf.groups}


def _timing_rows(md: Any, cfg: Config) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    tp = featurize_performance(md, cfg)
    off = np.clip(np.rint(tp.df["offset_ms"].to_numpy()), -OFFSET_HIST_MS, OFFSET_HIST_MS).astype(np.int64)
    hist = np.bincount(off + OFFSET_HIST_MS, minlength=2 * OFFSET_HIST_MS + 1).tolist()
    return tp.df, tp, {"target_hist": hist, "feature_groups": tp.groups}


def part_path(out_dir: Path, split: str, file_id: int) -> Path:
    return out_dir / f"split={split}" / f"part-{file_id:05d}.parquet"


def process_one(task: dict[str, Any]) -> dict[str, Any]:
    """Featurize one performance and write its Parquet part. Never raises."""
    cfg = _CFG
    assert cfg is not None
    info = {k: task[k] for k in ("file_id", "midi_filename", "split", "canonical_composer", "canonical_title", "year")}
    out = Path(task["out_path"])
    try:
        md = load_midi(task["midi_path"], include_drums=cfg.data.include_drums)
        df, pf, extra = (_timing_rows if _TASK == "timing" else _velocity_rows)(md, cfg)
        df.insert(0, "file_id", np.int32(task["file_id"]))

        info.update(
            n_notes=int(len(df)),
            duration_s=float(md.notes.offset.max()),
            beat_source=pf.grid.source,
            meter=pf.grid.meter_label,
            median_tempo_bpm=float(np.median(pf.grid.tempo_bpm)),
            **extra,
        )
        table = pa.Table.from_pandas(df, preserve_index=False)
        table = table.replace_schema_metadata({_INFO_KEY: json.dumps(info).encode()})
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.parent / f".{out.name}.tmp"  # dot-prefixed: ignored by pyarrow datasets
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, out)
    except Exception as exc:  # malformed file, empty file, ... -> skip gracefully
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def read_part_info(path: Path) -> dict[str, Any] | None:
    try:
        meta = pq.read_schema(path).metadata or {}
        return json.loads(meta[_INFO_KEY]) if _INFO_KEY in meta else None
    except Exception:
        return None


def build_tasks(root: Path, out_dir: Path, splits: list[str], limit: int | None, seed: int) -> list[dict[str, Any]]:
    meta = read_metadata(root).reset_index(drop=True)
    meta["file_id"] = np.arange(len(meta))  # row in the official CSV: stable across runs
    rng = np.random.default_rng(seed)
    tasks = []
    for split in splits:
        part = meta[meta["split"] == split].sort_values("midi_filename")
        if limit is not None and len(part) > limit:
            part = part.iloc[np.sort(rng.choice(len(part), size=limit, replace=False))]
        for row in part.itertuples(index=False):
            tasks.append(
                {
                    "file_id": int(row.file_id),
                    "midi_path": str(root / row.midi_filename),
                    "midi_filename": row.midi_filename,
                    "split": split,
                    "canonical_composer": row.canonical_composer,
                    "canonical_title": row.canonical_title,
                    "year": int(getattr(row, "year", 0) or 0),
                    "out_path": str(part_path(out_dir, split, int(row.file_id))),
                }
            )
    return tasks


def split_summary(files: list[dict[str, Any]], task: str) -> tuple[dict[str, Any], str]:
    stats: dict[str, Any] = {}
    rows = []
    name, origin = ("offset_ms", -OFFSET_HIST_MS) if task == "timing" else ("velocity", 0)
    for split in SPLITS:
        sel = [f for f in files if f["split"] == split]
        if not sel:
            continue
        hist = np.sum([f.get("target_hist", f.get("velocity_hist")) for f in sel], axis=0)  # parts from older builds
        s = stats_from_histogram(hist, origin)
        stats[split] = {"files": len(sel), "notes": int(s["n"]), name: s}
        rows.append({"split": split, "files": len(sel), "notes": int(s["n"]), **{k: v for k, v in s.items() if k != "n"}})
    table = format_table(
        rows, ["split", "files", "notes", "mean", "std", "min", "p5", "p25", "median", "p75", "p95", "max"], "{:.1f}"
    )
    return stats, table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maestro_dir", default=None, help="MAESTRO directory or maestro-v3.0.0-midi.zip")
    ap.add_argument("--output", default=None, help="output dataset directory (default: data.features_path)")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--limit_files", type=int, default=None, help="max files per split (random, seeded) for quick runs")
    ap.add_argument("--task", choices=["velocity", "timing"], default="velocity",
                    help="timing: reconstructed-score features + onset offsets (timing_features.py)")
    ap.add_argument("--quantize", action="store_true", help="shortcut for --set features.quantize=true")
    ap.add_argument("--overwrite", action="store_true", help="rebuild parts that already exist")
    add_config_args(ap)
    args = ap.parse_args()
    setup_logging()

    cfg = load_config(args.config, args.set)
    if args.quantize and args.task == "timing":
        ap.error("--quantize is the velocity pipeline's score-like experiment; the timing task always quantizes")
    if args.quantize:
        cfg = config_from_dict({"features": {"quantize": True}}, base=cfg)
    if args.maestro_dir:
        cfg = config_from_dict({"data": {"maestro_dir": args.maestro_dir}}, base=cfg)
    out_dir = Path(args.output or cfg.data.features_path)
    set_seed(cfg.train.seed)
    if cfg.features.context == "causal" and cfg.beat.source != "midi":
        log.warning(
            "causal context with beat.source=%s: the offline beat tracker itself looks ahead; "
            "use beat.source=midi for a strictly causal setup",
            cfg.beat.source,
        )

    root = locate_maestro(cfg.data.maestro_dir).resolve()
    cfg = config_from_dict({"data": {"maestro_dir": str(root)}}, base=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to mix parts built with different pipeline settings.
    build_cfg_path = out_dir / "_build_config.json"
    if args.task == "timing":
        pipeline = {"task": "timing", **cfg.timing_pipeline_dict(), "include_drums": cfg.data.include_drums}
    else:
        pipeline = {**cfg.pipeline_dict(), "include_drums": cfg.data.include_drums}
    if build_cfg_path.exists() and not args.overwrite and load_json(build_cfg_path) != json.loads(json.dumps(pipeline)):
        raise SystemExit(
            f"{out_dir} was built with a different pipeline config. Use --overwrite or a new --output."
        )
    save_json(pipeline, build_cfg_path)

    tasks = build_tasks(root, out_dir, args.splits, args.limit_files, cfg.train.seed)
    todo, infos = [], []
    for t in tasks:
        p = Path(t["out_path"])
        cached = None if args.overwrite or not p.exists() else read_part_info(p)
        if cached is not None:
            infos.append(cached)
        else:
            todo.append(t)
    log.info("MAESTRO root: %s", root)
    log.info("%d performances selected, %d cached, %d to process with %d workers", len(tasks), len(infos), len(todo), args.workers)

    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(cfg.to_dict(), args.task)) as ex:
            for info in tqdm(ex.map(process_one, todo, chunksize=2), total=len(todo), desc="featurizing", unit="file"):
                infos.append(info)
                if "error" in info:
                    log.warning("skipped %s: %s", info["midi_filename"], info["error"])
    log.info("feature extraction took %.1f s", time.time() - t0)

    ok = sorted((i for i in infos if "error" not in i), key=lambda i: i["file_id"])
    failures = [{"midi_filename": i["midi_filename"], "error": i["error"]} for i in infos if "error" in i]
    if not ok:
        raise SystemExit("no files were processed successfully")
    groups = ok[0]["feature_groups"]
    stats, table = split_summary(ok, args.task)
    timing = args.task == "timing"

    files = [{k: v for k, v in i.items() if k not in ("target_hist", "velocity_hist", "feature_groups")} for i in ok]
    save_json(
        {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": args.task,
            "maestro_dir": str(root),
            "config": cfg.to_dict(),
            "pipeline_config": cfg.timing_pipeline_dict() if timing else cfg.pipeline_dict(),
            "feature_columns": list(groups),
            "feature_groups": groups,
            "meta_columns": ["file_id", *META_COLUMNS, *(TIMING_META_COLUMNS if timing else [])],
            "target_columns": TIMING_TARGET_COLUMNS if timing else TARGET_COLUMNS,
            "performance_conditioned_columns": [] if timing else PC_COLUMNS,
            "files": files,
            "failures": failures,
            "split_stats": stats,
        },
        out_dir / "_meta.json",
    )

    unit = "onset offset vs reference beat, ms" if timing else "velocity"
    print(f"\nDataset summary (official MAESTRO split; whole performances per split; {unit})")
    print(table)
    print(f"\n{len(groups)} feature columns, {len(failures)} failed files -> {out_dir.resolve()}")
    if failures:
        for f in failures[:10]:
            print(f"  failed: {f['midi_filename']}: {f['error']}")


if __name__ == "__main__":
    main()
