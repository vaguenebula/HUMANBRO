"""Metrics for the micro-timing model (shared by train_timing.py and evaluate_timing.py).

Distribution quality: pinball loss per quantile (lower is better; its mean over the levels
approximates CRPS), calibration (how often the true offset falls below each predicted
quantile; should equal the level), and central-interval coverage and width (narrower is
better at the right coverage). The reference is the *unconditional* distribution: the
training-set quantiles applied to every note, which is what a "random humanize" knob does.

Median quality, split into the two things timing consists of:
* event timing: the mean offset of each onset group (is the chord early or late?);
* asynchrony: each note minus its group mean, for groups of 2+ notes (melody lead,
  rolled chords). ``top_lead_ms`` is the mean of that for the top note of each chord.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils import regression_metrics, within_piece_pearson


def pinball(y: np.ndarray, q: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Mean pinball loss per quantile level. ``q`` is (n, levels) or (levels,)."""
    q = np.broadcast_to(q, (len(y), len(alphas)))
    d = np.asarray(y, dtype=np.float64)[:, None] - q
    a = np.asarray(alphas)[None, :]
    return np.mean(np.maximum(a * d, (a - 1.0) * d), axis=0)


def coverage(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = np.broadcast_to(q, (len(y), q.shape[-1]))
    return np.mean(np.asarray(y)[:, None] <= q, axis=0)


def interval_rows(y: np.ndarray, q: np.ndarray, alphas: np.ndarray) -> list[dict[str, float]]:
    """Coverage and mean width of each symmetric central interval (a, 1 - a)."""
    q = np.broadcast_to(q, (len(y), len(alphas)))
    rows = []
    for i, a in enumerate(alphas):
        j = int(np.argmin(np.abs(alphas - (1.0 - a))))
        if a >= 0.5 or abs(alphas[j] - (1.0 - a)) > 1e-9:
            continue
        inside = (y >= q[:, i]) & (y <= q[:, j])
        rows.append({"interval": f"{100 * (1 - 2 * a):.0f}%", "nominal": 1 - 2 * a, "coverage": float(inside.mean()),
                     "mean_width_ms": float(np.mean(q[:, j] - q[:, i]))})
    return rows


def distribution_metrics(y: np.ndarray, q: np.ndarray, alphas: np.ndarray) -> dict[str, Any]:
    pb = pinball(y, q, alphas)
    return {
        "mean_pinball": float(pb.mean()),
        "pinball": dict(zip(map(str, alphas), pb.tolist())),
        "calibration": dict(zip(map(str, alphas), coverage(y, q).tolist())),
        "intervals": interval_rows(y, q, alphas),
    }


def group_keys(file_id: np.ndarray, group_id: np.ndarray) -> np.ndarray:
    return np.asarray(file_id, dtype=np.int64) * 10_000_000 + np.asarray(group_id, dtype=np.int64)


def median_metrics(y: np.ndarray, pred: np.ndarray, file_id: np.ndarray, group_id: np.ndarray,
                   is_top: np.ndarray) -> dict[str, Any]:
    """Point accuracy of one offset per note, overall and split into event timing vs asynchrony."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    key = group_keys(file_id, group_id)
    s = pd.DataFrame({"k": key, "y": y, "p": pred})
    grp = s.groupby("k", sort=False)
    y_ev, p_ev, size = grp["y"].transform("mean").to_numpy(), grp["p"].transform("mean").to_numpy(), grp["y"].transform("size").to_numpy()
    out: dict[str, Any] = {"all": regression_metrics(y, pred)}
    out["all"]["within_piece_pearson"] = within_piece_pearson(y, pred, file_id)
    first = ~pd.Series(key).duplicated().to_numpy()  # one row per onset group
    out["event"] = regression_metrics(y_ev[first], p_ev[first])
    multi = size > 1
    out["asynchrony"] = regression_metrics((y - y_ev)[multi], (pred - p_ev)[multi])
    top = multi & (np.asarray(is_top) > 0)
    out["top_lead_ms"] = {"true": float(np.mean((y - y_ev)[top])), "pred": float(np.mean((pred - p_ev)[top]))}
    return out


def realism_stats(v: np.ndarray, file_id: np.ndarray, group_id: np.ndarray) -> dict[str, float]:
    """Spread and continuity of one set of offsets (true, predicted median, or sampled).

    event_lag1_r: correlation of successive onset groups' mean offsets within a piece
    (how smoothly timing drifts); asynchrony_sd_ms: spread of notes around their chord.
    """
    v = np.asarray(v, dtype=np.float64)
    key = group_keys(file_id, group_id)
    s = pd.DataFrame({"f": np.asarray(file_id), "k": key, "v": v})
    grp = s.groupby("k", sort=False)
    ev = grp.agg(f=("f", "first"), v=("v", "mean"))
    lag1 = []
    for _, e in ev.groupby("f", sort=False):
        x = e["v"].to_numpy()
        if len(x) > 20 and x.std() > 0:
            lag1.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    within = v - grp["v"].transform("mean").to_numpy()
    multi = grp["v"].transform("size").to_numpy() > 1
    return {"sd_ms": float(np.std(v)), "event_sd_ms": float(ev["v"].std()),
            "event_lag1_r": float(np.nanmean(lag1)) if lag1 else float("nan"),
            "asynchrony_sd_ms": float(np.std(within[multi]))}
