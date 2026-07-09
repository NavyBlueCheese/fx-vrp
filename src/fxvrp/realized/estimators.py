"""Quadratic-variation estimators: RV, subsampled RV, TSRV, realised kernel,
realised semivariance.

Conventions: estimators act on intraday *log returns* (last axis) except TSRV,
which needs the underlying log-price grid to build its subsampled scales. All
quantities are in variance units over the sample window (no annualisation here;
conventions.md rule 3 keeps annualisation at the comparison layer).

Sources:
  - RV: Andersen, Bollerslev, Diebold & Labys (2003, Econometrica 71(2)).
  - TSRV: Zhang, Mykland & Aït-Sahalia (2005, JASA 100(472)), "two time scales"
    estimator with their small-sample adjustment (1 - n̄/n)^{-1}.
    # TODO(verify): confirm the adjusted-estimator equation number in ZMA (2005)
    # (the adjustment factor is stated in their §4; the test suite pins the
    # behaviour: noise-robust recovery of σ²T where naive RV diverges).
  - Realised kernel: Barndorff-Nielsen, Hansen, Lunde & Shephard (2008,
    Econometrica 76(6)) with the non-negative Parzen kernel and the bandwidth
    rule of their practice paper (2009, Econometrics Journal 12(3)):
    H* = c* ξ^{4/5} n^{3/5}, c* = 3.5134 for Parzen, ξ² = ω²/√(T ∫σ⁴ du),
    with ω² estimated from dense RV/(2n) and √(T∫σ⁴) proxied by sparse RV.
    # TODO(verify): c* = 3.5134 and the ξ² proxy against BNHLS (2009) eq. (15);
    # the ground-truth test only requires noise-robust recovery, which holds
    # for a wide band of H around the optimum.
  - Realised semivariance: Barndorff-Nielsen, Kinnebrock & Shephard (2010, in
    Volatility and Time Series Econometrics), RS± = Σ r² 1{r ≷ 0}.
"""

from __future__ import annotations

import numpy as np

from fxvrp._types import FloatArray

# Parzen kernel breakpoints and BNHLS (2009) bandwidth constant for Parzen
_PARZEN_KNEE = 0.5
_PARZEN_C_STAR = 3.5134
# structural minima: TSRV needs two scales; a bandwidth needs two returns
_MIN_SUBGRIDS = 2
_MIN_RETURNS = 2


def realized_variance(returns: FloatArray, axis: int = -1) -> FloatArray | float:
    """RV = Σ r². Consistent for QV of the log price as the grid refines."""
    result = np.sum(np.square(returns), axis=axis)
    return float(result) if np.ndim(result) == 0 else np.asarray(result, dtype=np.float64)


def subsampled_rv(log_prices: FloatArray, stride: int) -> float:
    """Average RV over the ``stride`` offset subgrids of a log-price series.

    Subsample-averaging (Zhang et al. 2005, §2) uses all data at a coarse scale
    instead of discarding all but one offset grid.
    """
    if log_prices.ndim != 1:
        raise ValueError("log_prices must be one-dimensional")
    if stride < 1 or stride >= log_prices.size:
        raise ValueError(f"stride must be in [1, n_prices); got {stride}")
    rvs = [float(np.sum(np.diff(log_prices[offset::stride]) ** 2)) for offset in range(stride)]
    return float(np.mean(rvs))


def tsrv(log_prices: FloatArray, n_subgrids: int) -> float:
    """Two-scale realised variance (Zhang, Mykland & Aït-Sahalia 2005).

    TSRV = (1 - n̄/n)^{-1} [ (1/K) Σ_k RV^{(k)} - (n̄/n) RV^{(all)} ],
    n̄ = (n - K + 1)/K, with K = ``n_subgrids`` and n the number of full-grid
    returns. The slow-scale average is contaminated by n̄·2ω² of noise, the fast
    scale by n·2ω²; the linear combination cancels the noise exactly in
    expectation, and the leading factor undoes the resulting downward bias in
    the signal term.
    """
    if log_prices.ndim != 1:
        raise ValueError("log_prices must be one-dimensional")
    n = log_prices.size - 1
    k = n_subgrids
    if k < _MIN_SUBGRIDS or k >= n:
        raise ValueError(f"n_subgrids must be in [2, n_returns); got {k}")

    slow = np.mean([float(np.sum(np.diff(log_prices[offset::k]) ** 2)) for offset in range(k)])
    fast = float(np.sum(np.diff(log_prices) ** 2))
    n_bar = (n - k + 1) / k
    unadjusted = slow - (n_bar / n) * fast
    return float(unadjusted / (1.0 - n_bar / n))


def parzen_kernel(x: FloatArray) -> FloatArray:
    """Parzen weight function on [0, 1]; zero outside."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(x)
    near = x <= _PARZEN_KNEE
    far = (x > _PARZEN_KNEE) & (x <= 1.0)
    out[near] = 1.0 - 6.0 * x[near] ** 2 + 6.0 * x[near] ** 3
    out[far] = 2.0 * (1.0 - x[far]) ** 3
    return out


def kernel_bandwidth(returns: FloatArray, sparse_stride: int) -> int:
    """BNHLS (2009) plug-in bandwidth H* = c* ξ^{4/5} n^{3/5} for the Parzen kernel.

    ω² (noise variance) is estimated as RV_dense / (2n) — the standard
    Bandi-Russell / ZMA noise estimator — and √(T ∫σ⁴) is proxied by a sparse,
    noise-insensitive RV.
    """
    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    n = returns.size
    if n < _MIN_RETURNS:
        raise ValueError("need at least two returns")
    sparse_stride = max(1, min(sparse_stride, n // 2))

    omega_sq = float(np.sum(returns**2)) / (2.0 * n)
    prices = np.concatenate([[0.0], np.cumsum(returns)])
    iv_proxy = (
        subsampled_rv(prices, sparse_stride) if sparse_stride > 1 else float(np.sum(returns**2))
    )
    if iv_proxy <= 0.0:
        return 1
    xi_sq = omega_sq / iv_proxy
    h_star = _PARZEN_C_STAR * xi_sq ** (2.0 / 5.0) * n ** (3.0 / 5.0)
    return max(1, int(np.ceil(h_star)))


def realized_kernel(returns: FloatArray, bandwidth: int) -> float:
    """Realised kernel (BNHLS 2008) with Parzen weights:

    RK = Σ_{h=-H}^{H} k(h/(H+1)) γ_h,   γ_h = Σ_i r_i r_{i-|h|}.

    The Parzen kernel is smooth and non-negative-definite, so RK cannot go
    negative asymptotically; end-point jittering from the practice paper is not
    implemented (documented simplification — bias is O(1/n) at our sample sizes).
    """
    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    n = returns.size
    if bandwidth < 0 or bandwidth >= n:
        raise ValueError(f"bandwidth must be in [0, n_returns); got {bandwidth}")

    gamma_0 = float(np.dot(returns, returns))
    total = gamma_0
    for h in range(1, bandwidth + 1):
        gamma_h = float(np.dot(returns[h:], returns[:-h]))
        weight = parzen_kernel(np.asarray([h / (bandwidth + 1)])).item()
        total += 2.0 * weight * gamma_h
    return total


def realized_semivariance(returns: FloatArray, axis: int = -1) -> tuple[FloatArray, FloatArray]:
    """(RS⁺, RS⁻): squared returns split by sign (BNKS 2010).

    RS⁺ + RS⁻ = RV identically; RS⁻ loads on negative jumps, making the signed
    jump variation ΔJ = RS⁺ - RS⁻ the crash-risk conditioning variable.
    """
    squared = np.square(returns)
    rs_plus = np.sum(np.where(returns > 0.0, squared, 0.0), axis=axis)
    rs_minus = np.sum(np.where(returns < 0.0, squared, 0.0), axis=axis)
    return (
        np.asarray(rs_plus, dtype=np.float64),
        np.asarray(rs_minus, dtype=np.float64),
    )
