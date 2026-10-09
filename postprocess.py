"""Light post-processing of predicted velocities.

Smoothing works on onset groups, not on individual notes:
* each group's mean velocity is smoothed with a zero-phase (forward+backward)
  exponential moving average over successive groups, which removes
  note-to-note jitter along the melodic/temporal line without lag;
* every note keeps its offset from its group mean, so chord voicing
  (e.g. a louder top note) is untouched;
* groups that stand out above the smoothed trend by more than
  ``accent_threshold`` are treated as accents and left as predicted.
"""

from __future__ import annotations

import numpy as np

from config import SmoothingConfig


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def smooth_velocities(
    velocity: np.ndarray, group_id: np.ndarray, alpha: float, strength: float, accent_threshold: float
) -> np.ndarray:
    """``velocity`` and ``group_id`` in canonical order (group ids 0..G-1, non-decreasing)."""
    v = np.asarray(velocity, dtype=np.float64)
    g = np.asarray(group_id, dtype=np.int64)
    if len(v) == 0 or strength <= 0:
        return v.copy()
    n_groups = int(g.max()) + 1
    counts = np.bincount(g, minlength=n_groups)
    gmean = np.bincount(g, weights=v, minlength=n_groups) / np.maximum(counts, 1)
    trend = 0.5 * (_ema(gmean, alpha) + _ema(gmean[::-1], alpha)[::-1])
    smoothed = gmean + strength * (trend - gmean)
    accents = (gmean - trend) > accent_threshold
    smoothed[accents] = gmean[accents]
    return v + (smoothed - gmean)[g]


def apply_smoothing(velocity: np.ndarray, group_id: np.ndarray, cfg: SmoothingConfig) -> np.ndarray:
    if not cfg.enabled:
        return np.asarray(velocity, dtype=np.float64)
    return smooth_velocities(velocity, group_id, cfg.alpha, cfg.strength, cfg.accent_threshold)


def shape_dynamics(velocity: np.ndarray, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Expand/compress dynamics around the piece median, then shift the overall level."""
    v = np.asarray(velocity, dtype=np.float64)
    centre = np.median(v) if len(v) else 0.0
    return centre + scale * (v - centre) + offset
