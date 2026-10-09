"""Shared helpers: logging, seeding, device detection, metrics, feature-cache I/O."""

from __future__ import annotations

import json
import logging
import os
import random
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
FEATURES_META_NAME = "_meta.json"


def setup_logging(level: str | int = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(preference: str = "auto") -> str:
    """'cuda' if requested/available and XGBoost can actually train on it, else 'cpu'."""
    if preference == "cpu":
        return "cpu"
    import xgboost as xgb

    if not xgb.build_info().get("USE_CUDA", False):
        if preference == "cuda":
            log.warning("this XGBoost build has no CUDA support; using CPU")
        return "cpu"
    try:
        x = np.random.default_rng(0).random((64, 3)).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # XGBoost warns (not raises) when it falls back
            xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0}, xgb.DMatrix(x, label=x[:, 0]), 1)
        return "cuda"
    except Exception as exc:  # no driver / no visible GPU / fallback warning
        if preference == "cuda":
            log.warning("CUDA requested but unusable (%s); using CPU", exc)
        return "cpu"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    err = p - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),
        "pearson": _pearson(y, p),
        "spearman": float("nan"),
    }
    try:
        from scipy.stats import spearmanr

        if np.ptp(p) > 0 and np.ptp(y) > 0:
            out["spearman"] = float(spearmanr(y, p).statistic)
    except ImportError:
        pass
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:  # correlation undefined for constants
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def within_piece_pearson(y_true: np.ndarray, y_pred: np.ndarray, file_ids: np.ndarray, min_notes: int = 50) -> float:
    """Mean per-file correlation: does the model get the *shape* of each performance right?

    Unaffected by per-performance loudness offsets (recording session, piano
    calibration, player), which no score-based model can know.
    """
    df = pd.DataFrame({"y": y_true, "p": y_pred, "f": file_ids})
    vals = [_pearson(g["y"].to_numpy(), g["p"].to_numpy()) for _, g in df.groupby("f") if len(g) >= min_notes]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def velocity_stats(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    q = np.percentile(v, [5, 25, 50, 75, 95]) if len(v) else [np.nan] * 5
    return {
        "n": int(len(v)),
        "mean": float(v.mean()) if len(v) else float("nan"),
        "std": float(v.std()) if len(v) else float("nan"),
        "min": float(v.min()) if len(v) else float("nan"),
        "p5": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "max": float(v.max()) if len(v) else float("nan"),
    }


def stats_from_histogram(hist: np.ndarray, origin: float = 0.0) -> dict[str, float]:
    """velocity_stats for data summarised as a unit-bin histogram (bin i holds value i + origin)."""
    hist = np.asarray(hist, dtype=np.float64)
    n = hist.sum()
    if n == 0:
        return velocity_stats(np.array([]))
    vals = np.arange(len(hist)) + origin
    mean = float((hist * vals).sum() / n)
    std = float(np.sqrt((hist * (vals - mean) ** 2).sum() / n))
    cdf = np.cumsum(hist) / n
    pct = {f"p{q}": float(vals[np.searchsorted(cdf, q / 100.0)]) for q in (5, 25, 50, 75, 95)}
    nz = np.flatnonzero(hist) + origin
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "min": float(nz[0]),
        "p5": pct["p5"],
        "p25": pct["p25"],
        "median": pct["p50"],
        "p75": pct["p75"],
        "p95": pct["p95"],
        "max": float(nz[-1]),
    }


def format_table(rows: list[dict[str, Any]], columns: list[str], floatfmt: str = "{:.3f}") -> str:
    """Plain markdown table."""

    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "nan" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)

    cells = [[fmt(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c) for i, c in enumerate(columns)]
    line = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in cells]
    return "\n".join([line, sep, *body])


def split_stats_table(df: pd.DataFrame) -> str:
    rows = []
    for split in SPLITS:
        part = df[df["split"] == split]
        if part.empty:
            continue
        s = velocity_stats(part["velocity"].to_numpy())
        rows.append({"split": split, "files": part["file_id"].nunique(), "notes": len(part), **s})
    return format_table(
        rows, ["split", "files", "notes", "mean", "std", "min", "p5", "p25", "median", "p75", "p95", "max"], "{:.1f}"
    )


# ---------------------------------------------------------------------------
# JSON + feature-cache I/O
# ---------------------------------------------------------------------------


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def features_meta_path(data_path: str | Path) -> Path:
    return Path(data_path) / FEATURES_META_NAME


def load_features_meta(data_path: str | Path) -> dict[str, Any]:
    p = features_meta_path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run build_dataset.py first")
    return load_json(p)


def choose_files(
    files: list[dict[str, Any]], splits: Iterable[str], max_per_split: int | None, seed: int
) -> list[int]:
    """File ids for the requested splits, optionally subsampled *by file* (never by note)."""
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for split in splits:
        ids = sorted(f["file_id"] for f in files if f["split"] == split)
        if max_per_split is not None and len(ids) > max_per_split:
            ids = sorted(rng.choice(ids, size=max_per_split, replace=False).tolist())
        chosen.extend(ids)
    return chosen


def split_part_files(data_path: str | Path, split: str, file_ids: Iterable[int] | None = None) -> list[Path]:
    """Parquet parts of one split, ordered by file id (deterministic row order)."""
    parts = sorted((Path(data_path) / f"split={split}").glob("part-*.parquet"))
    if file_ids is not None:
        wanted = {int(i) for i in file_ids}
        parts = [p for p in parts if int(p.stem.split("-")[1]) in wanted]
    return parts


def load_split_matrix(
    data_path: str | Path,
    split: str,
    feature_columns: list[str],
    extra_columns: list[str],
    file_ids: Iterable[int] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Memory-lean loader for training: one preallocated float32 (rows x features) matrix.

    Parts are read one performance at a time, so peak memory is the matrix itself
    plus one small file, instead of several full-size pandas copies.
    """
    import pyarrow.parquet as pq

    parts = split_part_files(data_path, split, file_ids)
    counts = [pq.ParquetFile(p).metadata.num_rows for p in parts]
    x = np.empty((sum(counts), len(feature_columns)), dtype=np.float32)
    extras: dict[str, list[np.ndarray]] = {c: [] for c in extra_columns}
    pos = 0
    for path, k in zip(parts, counts):
        table = pq.read_table(path, columns=list(dict.fromkeys([*feature_columns, *extra_columns])))
        for j, c in enumerate(feature_columns):
            x[pos : pos + k, j] = table.column(c).to_numpy()  # nulls -> NaN (XGBoost "missing")
        for c in extra_columns:
            extras[c].append(table.column(c).to_numpy())
        pos += k
    info = pd.DataFrame({c: np.concatenate(v) if v else np.array([]) for c, v in extras.items()})
    info["split"] = split
    return x, info


def load_feature_table(
    data_path: str | Path,
    columns: list[str],
    splits: Iterable[str] | None = None,
    file_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Read only the requested columns/splits/files from the partitioned Parquet cache."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(data_path), format="parquet", partitioning="hive")
    filt = None
    if splits is not None:
        filt = ds.field("split").isin(list(splits))
    if file_ids is not None:
        f2 = ds.field("file_id").isin([int(i) for i in file_ids])
        filt = f2 if filt is None else filt & f2
    cols = list(dict.fromkeys([*columns, "split"]))
    table = dataset.to_table(columns=cols, filter=filt)
    df = table.to_pandas()
    df["split"] = df["split"].astype(str)
    sort_cols = [c for c in ("file_id", "note_idx") if c in df.columns]
    if sort_cols:  # fragments are read in parallel; restore a deterministic order
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df
