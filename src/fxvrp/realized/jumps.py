"""Jump-robust variation measures and the BNS/Huang-Tauchen jump test.

Sources:
  - Bipower variation: Barndorff-Nielsen & Shephard (2004, J. Financial
    Econometrics 2(1)): BPV = μ₁⁻² Σ|r_i||r_{i-1}| with μ₁ = E|Z| = √(2/π),
    consistent for the *continuous* part ∫σ² du under finite-activity jumps.
    The n/(n-1) small-sample factor follows Huang & Tauchen (2005).
  - Tripower quarticity: TQ = n μ_{4/3}⁻³ (n/(n-2)) Σ |r_i r_{i-1} r_{i-2}|^{4/3},
    μ_{4/3} = 2^{2/3} Γ(7/6)/Γ(1/2), consistent for (∫σ² du scale) ∫σ⁴ du · T
    so that TQ/BPV² → 1 under constant volatility.
  - Ratio jump test: Huang & Tauchen (2005, J. Financial Econometrics 3(4)),
    the ratio statistic z_{TP,rm}:

        z = [(RV - BPV)/RV] / sqrt( θ (1/n) max(1, TQ/BPV²) ),
        θ = (π/2)² + π - 5 ≈ 0.6090,

    asymptotically N(0,1) under the no-jump null.
    # TODO(verify): θ and the max(1, ·) truncation against Huang & Tauchen
    # (2005) — the brief itself flags that the denominator constant varies
    # across the literature. The test suite pins size under H₀ (GBM) and power
    # under H₁ (Merton) empirically, which is the property we actually rely on.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from fxvrp._types import FloatArray

_MU_1 = float(np.sqrt(2.0 / np.pi))  # E|Z|
_MU_43 = float(2.0 ** (2.0 / 3.0) * special.gamma(7.0 / 6.0) / special.gamma(0.5))
_THETA = (np.pi / 2.0) ** 2 + np.pi - 5.0
# structural minima of the estimator definitions (adjacent-return products)
_MIN_RETURNS_BIPOWER = 2
_MIN_RETURNS_TRIPOWER = 3


def bipower_variation(returns: FloatArray, axis: int = -1) -> FloatArray | float:
    """BPV: jump-robust estimator of the continuous quadratic variation."""
    r = np.moveaxis(np.asarray(returns, dtype=np.float64), axis, -1)
    n = r.shape[-1]
    if n < _MIN_RETURNS_BIPOWER:
        raise ValueError("bipower variation needs at least two returns")
    abs_r = np.abs(r)
    bpv = _MU_1**-2 * (n / (n - 1)) * np.sum(abs_r[..., 1:] * abs_r[..., :-1], axis=-1)
    return float(bpv) if np.ndim(bpv) == 0 else np.asarray(bpv, dtype=np.float64)


def tripower_quarticity(returns: FloatArray, axis: int = -1) -> FloatArray | float:
    """TQ: jump-robust estimator scaled so TQ/BPV² → 1 under constant volatility."""
    r = np.moveaxis(np.asarray(returns, dtype=np.float64), axis, -1)
    n = r.shape[-1]
    if n < _MIN_RETURNS_TRIPOWER:
        raise ValueError("tripower quarticity needs at least three returns")
    abs_r = np.abs(r) ** (4.0 / 3.0)
    core = np.sum(abs_r[..., 2:] * abs_r[..., 1:-1] * abs_r[..., :-2], axis=-1)
    tq = n * _MU_43**-3 * (n / (n - 2)) * core
    return float(tq) if np.ndim(tq) == 0 else np.asarray(tq, dtype=np.float64)


def jump_variation(returns: FloatArray, axis: int = -1) -> FloatArray | float:
    """JV = max(RV - BPV, 0): the jump part of quadratic variation."""
    rv = np.sum(np.square(returns), axis=axis)
    bpv = bipower_variation(returns, axis=axis)
    jv = np.maximum(np.asarray(rv) - np.asarray(bpv), 0.0)
    return float(jv) if np.ndim(jv) == 0 else np.asarray(jv, dtype=np.float64)


def bns_test_statistic(returns: FloatArray) -> float:
    """Huang-Tauchen ratio jump statistic; ~N(0,1) under the no-jump null.

    Large positive values indicate that RV exceeds BPV by more than sampling
    noise explains — i.e. jumps.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    n = r.size
    if n < _MIN_RETURNS_TRIPOWER:
        raise ValueError("the jump test needs at least three returns")

    rv = float(np.sum(r**2))
    if rv <= 0.0:
        return 0.0
    bpv = float(bipower_variation(r))
    tq = float(tripower_quarticity(r))
    ratio = (rv - bpv) / rv
    quarticity_guard = max(1.0, tq / bpv**2) if bpv > 0.0 else 1.0
    return float(ratio / np.sqrt(_THETA * quarticity_guard / n))
