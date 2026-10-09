"""Turn predicted offset quantiles into sampled micro-timing.

Per note, the model predicts quantiles q_a of the onset offset at levels a in A. The
note's quantile function Q is the piecewise-linear interpolation of those quantiles on
the normal-score scale (knots at Phi^-1(a)), extended linearly past the outer knots. A
sample is ``Q(z)`` for a standard-normal ``z``; ``z = 0`` gives the median.

Independent ``z`` per note would be white jitter: chords smeared at random, no
continuity from one beat to the next. Real deviations from the model's expectation are
correlated, so ``z`` comes from a Gaussian copula fitted on validation data, using the
normal scores of the true offsets under the predicted distributions (``normal_scores``):

    z_i = sqrt(rho) * a_g(i) + sqrt(1 - rho) * e_i            e_i ~ N(0, 1), independent
    a_g = phi_g * a_g-1 + sqrt(1 - phi_g^2) * eta_g          phi_g = exp(-dt_g / ell)

``a_g`` is shared by the notes of one onset group (chord), so ``rho`` is the within-chord
correlation, and the shared part decays with the distance ``dt_g`` (beats) between
successive groups, with correlation length ``ell``. Every ``z_i`` is still N(0, 1).

Marginal recalibration: if the predicted distributions are slightly too narrow or wide,
the validation normal scores are not exactly N(0, 1). Their empirical quantiles are stored
(``calib_u`` -> ``calib_z``), and each draw is mapped through them before it reaches the
quantile function, so samples match the spread actually observed on held-out
performances. The copula is fitted on the recalibrated scores. ``temperature`` scales
``z`` before that mapping (0 = the middle of each note's distribution, 1 = calibrated spread).

Random numbers come from ``PortableRng`` (SplitMix64 + Box-Muller), which the C++ runtime
implements identically, so a seed gives the same take in Python and C++.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import ndtr, ndtri

Z_MAX = 2.5  # no draws beyond ~0.6% / 99.4%: linear tails are a guess out there

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


class PortableRng:
    """SplitMix64 uniforms and Box-Muller normals, mirrored by cpp/src/timing.cpp.

    Uniform k (k = 1, 2, ...) is mix(seed + k * gamma) mapped to ((z >> 11) + 0.5) / 2^53.
    Normals come in (cos, sin) pairs from two consecutive uniforms; an odd request drops
    the last sine, so the stream position after a call depends only on the counts drawn.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.count = 0

    @staticmethod
    def new_seed() -> int:
        import secrets

        return secrets.randbits(63)

    def uniforms(self, n: int) -> np.ndarray:
        k = np.arange(self.count + 1, self.count + n + 1, dtype=np.uint64)
        self.count += n
        with np.errstate(over="ignore"):
            z = np.uint64(self.seed) + k * _GAMMA
            z = (z ^ (z >> np.uint64(30))) * _MIX1
            z = (z ^ (z >> np.uint64(27))) * _MIX2
            z = z ^ (z >> np.uint64(31))
        return ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)

    def normals(self, n: int) -> np.ndarray:
        m = (n + 1) // 2
        u = self.uniforms(2 * m)
        r = np.sqrt(-2.0 * np.log(u[0::2]))
        t = 2.0 * np.pi * u[1::2]
        out = np.empty(2 * m)
        out[0::2] = r * np.cos(t)
        out[1::2] = r * np.sin(t)
        return out[:n]


@dataclass
class CopulaParams:
    rho: float = 0.5  # correlation of the normal scores of two notes in one chord
    ell: float = 1.0  # correlation length of the shared part, in beats
    # Marginal recalibration: validation normal scores at these probability levels.
    calib_u: list[float] | None = None
    calib_z: list[float] | None = None
    # Binned evidence the fit came from (lag in beats -> correlation), for reports.
    lag_beats: list[float] | None = None
    lag_corr: list[float] | None = None
    n_pairs_within: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> CopulaParams:
        return cls(**d) if d else cls()


def normal_knots(alphas: np.ndarray) -> np.ndarray:
    a = np.asarray(alphas, dtype=np.float64)
    if np.any(np.diff(a) <= 0) or a[0] <= 0 or a[-1] >= 1:
        raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
    return ndtri(a)


def sort_quantiles(q: np.ndarray) -> np.ndarray:
    """Fix quantile crossing by rearrangement (sorting each row), which never increases pinball loss."""
    return np.sort(np.asarray(q, dtype=np.float64), axis=1)


def quantile_function(q: np.ndarray, knots: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Q_i(z_i) for every row: piecewise linear in z between knots, linear past the ends."""
    z = np.clip(np.asarray(z, dtype=np.float64), -3.5, 3.5)
    k = np.clip(np.searchsorted(knots, z) - 1, 0, len(knots) - 2)
    rows = np.arange(len(q))
    lo, hi = q[rows, k], q[rows, k + 1]
    t = (z - knots[k]) / (knots[k + 1] - knots[k])
    return lo + t * (hi - lo)


def normal_scores(q: np.ndarray, knots: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Inverse of ``quantile_function``: the z with Q_i(z) = y_i (probability integral transform)."""
    y = np.asarray(y, dtype=np.float64)
    k = np.clip((q < y[:, None]).sum(axis=1) - 1, 0, len(knots) - 2)
    rows = np.arange(len(q))
    lo, hi = q[rows, k], q[rows, k + 1]
    width = hi - lo
    t = np.where(width > 1e-9, (y - lo) / np.where(width > 1e-9, width, 1.0), 0.5)
    return np.clip(knots[k] + t * (knots[k + 1] - knots[k]), -3.5, 3.5)


# ---------------------------------------------------------------------------
# Copula + marginal recalibration: fit on validation data, sample at inference
# ---------------------------------------------------------------------------

CALIB_U = np.linspace(0.0025, 0.9975, 101)


def to_model_scores(z: np.ndarray, params: CopulaParams) -> np.ndarray:
    """Calibrated N(0, 1) score -> the model's own normal-score scale (identity without calibration)."""
    if not params.calib_u:
        return np.asarray(z, dtype=np.float64)
    return np.interp(ndtr(z), params.calib_u, params.calib_z)


def calibrated_scores(z_model: np.ndarray, params: CopulaParams) -> np.ndarray:
    """Inverse of ``to_model_scores``."""
    if not params.calib_u:
        return np.asarray(z_model, dtype=np.float64)
    u = np.interp(z_model, params.calib_z, params.calib_u)
    return ndtri(np.clip(u, 1e-6, 1 - 1e-6))


def fit_sampler(z_model: np.ndarray, file_id: np.ndarray, group_id: np.ndarray, group_beat: np.ndarray,
                seed: int = 0) -> CopulaParams:
    """Marginal recalibration from validation normal scores, then the copula on the recalibrated scores."""
    calib_z = np.quantile(np.asarray(z_model, dtype=np.float64), CALIB_U)
    calib_z = np.maximum.accumulate(calib_z + np.arange(len(calib_z)) * 1e-9)  # strictly increasing
    base = CopulaParams(calib_u=CALIB_U.tolist(), calib_z=calib_z.tolist())
    params = fit_copula(calibrated_scores(z_model, base), file_id, group_id, group_beat, seed=seed)
    params.calib_u, params.calib_z = base.calib_u, base.calib_z
    return params


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 30:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit_copula(
    z: np.ndarray,
    file_id: np.ndarray,
    group_id: np.ndarray,
    group_beat: np.ndarray,
    max_lag_groups: int = 12,
    seed: int = 0,
) -> CopulaParams:
    """Estimate rho (within-chord) and ell (decay in beats) from normal scores in canonical order.

    rho: correlation of adjacent chord-mates. ell: one random note per onset group, pairs of
    groups up to ``max_lag_groups`` apart, correlations binned by beat distance, then a least
    squares fit of rho * exp(-d / ell).
    """
    z = np.asarray(z, dtype=np.float64)
    f = np.asarray(file_id)
    g = np.asarray(group_id)
    same = (f[1:] == f[:-1]) & (g[1:] == g[:-1])
    rho = _corr(z[:-1][same], z[1:][same])
    rho = float(np.clip(rho if np.isfinite(rho) else 0.5, 0.0, 0.99))

    # One random representative per (file, group): its z and its group's beat position.
    rng = np.random.default_rng(seed)
    starts = np.flatnonzero(np.concatenate([[True], ~same]))
    sizes = np.diff(np.concatenate([starts, [len(z)]]))
    pick = starts + (rng.random(len(starts)) * sizes).astype(np.int64)
    rz, rb, rf = z[pick], np.asarray(group_beat, dtype=np.float64)[pick], f[pick]

    edges = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    sums = np.zeros((len(edges) - 1, 6))  # n, sum a, sum b, sum ab, sum a2+b2 (pooled), sum distance
    for lag in range(1, max_lag_groups + 1):
        ok = rf[lag:] == rf[:-lag]
        a, b = rz[:-lag][ok], rz[lag:][ok]
        d = (rb[lag:] - rb[:-lag])[ok]
        k = np.searchsorted(edges, d, side="right") - 1
        valid = (k >= 0) & (k < len(edges) - 1)
        for col, vals in enumerate((np.ones(valid.sum()), a[valid], b[valid], a[valid] * b[valid],
                                    a[valid] ** 2 + b[valid] ** 2, d[valid])):
            sums[:, col] += np.bincount(k[valid], weights=vals, minlength=len(edges) - 1)
    n = sums[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a, mean_b = sums[:, 1] / n, sums[:, 2] / n
        cov = sums[:, 3] / n - mean_a * mean_b
        var = sums[:, 4] / (2 * n) - ((mean_a + mean_b) / 2) ** 2
        corr = cov / var
        centres = sums[:, 5] / n  # mean distance of the pairs in each bin
    use = (n >= 200) & np.isfinite(corr)
    ells = np.geomspace(0.05, 64.0, 400)
    if use.sum() >= 2 and rho > 0.01:
        pred = rho * np.exp(-centres[use][None, :] / ells[:, None])
        err = ((pred - corr[use][None, :]) ** 2 * n[use][None, :]).sum(axis=1)
        ell = float(ells[int(np.argmin(err))])
    else:
        ell = 1.0
    return CopulaParams(rho=rho, ell=ell, lag_beats=centres[use].tolist(), lag_corr=corr[use].tolist(),
                        n_pairs_within=int(same.sum()))


def sample_latent(group_id: np.ndarray, group_beat: np.ndarray, params: CopulaParams, rng: PortableRng) -> np.ndarray:
    """Standard-normal scores for one piece (canonical order), correlated as described above."""
    g = np.asarray(group_id, dtype=np.int64)
    n_groups = int(g.max()) + 1 if len(g) else 0
    beat_of_group = np.zeros(n_groups)
    beat_of_group[g] = np.asarray(group_beat, dtype=np.float64)
    dt = np.diff(beat_of_group, prepend=beat_of_group[0] if n_groups else 0.0)
    phi = np.exp(-np.maximum(dt, 0.0) / max(params.ell, 1e-6))
    eta = rng.normals(n_groups)
    shared = np.empty(n_groups)
    acc = eta[0] if n_groups else 0.0
    for k in range(n_groups):
        if k:
            acc = phi[k] * acc + np.sqrt(1.0 - phi[k] ** 2) * eta[k]
        shared[k] = acc
    rho = float(np.clip(params.rho, 0.0, 1.0))
    return np.sqrt(rho) * shared[g] + np.sqrt(1.0 - rho) * rng.normals(len(g))


def sample_offsets(
    q: np.ndarray,
    alphas: np.ndarray,
    group_id: np.ndarray,
    group_beat: np.ndarray,
    params: CopulaParams,
    temperature: float = 1.0,
    rng: PortableRng | None = None,
) -> np.ndarray:
    """One draw of offsets (same unit as ``q``) for one piece in canonical order."""
    rng = rng or PortableRng(PortableRng.new_seed())
    z = np.clip(temperature * sample_latent(group_id, group_beat, params, rng), -Z_MAX, Z_MAX)
    return quantile_function(sort_quantiles(q), normal_knots(alphas), to_model_scores(z, params))
