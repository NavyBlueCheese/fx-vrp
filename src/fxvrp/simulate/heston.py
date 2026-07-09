"""Heston (1993) stochastic volatility with full-truncation Euler stepping.

Model:
    dS_t = mu S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,   d<W^S, W^v>_t = rho dt

Scheme: full truncation Euler of Lord, Koekkoek & van Dijk (2010, *Quantitative
Finance* 10(2), scheme "FT" in their Table 1): the variance argument enters both
drift and diffusion as v^+ = max(v, 0), which they show has the smallest positive
bias among the Euler fixes. The variance path itself may go negative and is floored
only inside the coefficients; the *recorded* spot variance is v^+.

Ground truth recorded per path: the integrated variance IV = ∫_0^T v_u^+ du
(left-Riemann sum on the simulation grid). Downstream realised-variance estimators
must recover this path-specific quantity, not its expectation.

The expectation itself is analytic for the CIR variance process:
    E[∫_0^T v_u du] = theta T + (v0 - theta) (1 - e^{-kappa T}) / kappa,
used as an independent cross-check (Cox, Ingersoll & Ross 1985 mean, integrated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fxvrp._types import FloatArray


@dataclass(frozen=True)
class HestonParams:
    kappa: float  # mean-reversion speed of variance
    theta: float  # long-run variance
    xi: float  # vol of vol
    rho: float  # spot-variance correlation
    v0: float  # initial variance

    def validate(self) -> None:
        if self.kappa <= 0.0 or self.theta <= 0.0 or self.xi <= 0.0:
            raise ValueError("kappa, theta and xi must be positive")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must lie in [-1, 1], got {self.rho}")
        if self.v0 < 0.0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")

    @property
    def feller_satisfied(self) -> bool:
        """Feller condition 2 kappa theta >= xi^2 (variance a.s. positive)."""
        return 2.0 * self.kappa * self.theta >= self.xi**2


@dataclass(frozen=True)
class HestonPaths:
    times: FloatArray  # (n_steps + 1,)
    log_prices: FloatArray  # (n_paths, n_steps + 1)
    variances: FloatArray  # (n_paths, n_steps + 1), spot variance v^+
    integrated_variance: FloatArray  # (n_paths,), ∫ v^+ dt over the horizon
    params: HestonParams

    @property
    def prices(self) -> FloatArray:
        return np.exp(self.log_prices)

    @property
    def horizon(self) -> float:
        return float(self.times[-1])

    def log_returns(self) -> FloatArray:
        return np.diff(self.log_prices, axis=1)


def expected_integrated_variance(params: HestonParams, horizon: float) -> float:
    """E[∫_0^T v_u du] for the CIR variance process (exact)."""
    kappa, theta, v0 = params.kappa, params.theta, params.v0
    return float(theta * horizon + (v0 - theta) * (1.0 - np.exp(-kappa * horizon)) / kappa)


def simulate_heston(
    *,
    s0: float,
    mu: float,
    params: HestonParams,
    horizon: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> HestonPaths:
    """Simulate correlated (log S, v) paths with the full-truncation Euler scheme."""
    params.validate()
    if s0 <= 0.0:
        raise ValueError(f"s0 must be positive, got {s0}")
    if horizon <= 0.0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be at least 1")

    dt = horizon / n_steps
    sqrt_dt = np.sqrt(dt)

    log_prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variances = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_prices[:, 0] = np.log(s0)
    variances[:, 0] = params.v0
    integrated = np.zeros(n_paths, dtype=np.float64)

    v_raw = np.full(n_paths, params.v0, dtype=np.float64)
    corr_orth = np.sqrt(1.0 - params.rho**2)

    for step in range(n_steps):
        v_plus = np.maximum(v_raw, 0.0)
        z_v = rng.standard_normal(n_paths)
        z_orth = rng.standard_normal(n_paths)
        z_s = params.rho * z_v + corr_orth * z_orth

        integrated += v_plus * dt  # left-Riemann sum of v^+
        log_prices[:, step + 1] = (
            log_prices[:, step] + (mu - v_plus / 2.0) * dt + np.sqrt(v_plus) * sqrt_dt * z_s
        )
        v_raw = (
            v_raw
            + params.kappa * (params.theta - v_plus) * dt
            + params.xi * np.sqrt(v_plus) * sqrt_dt * z_v
        )
        variances[:, step + 1] = np.maximum(v_raw, 0.0)

    times = np.linspace(0.0, horizon, n_steps + 1, dtype=np.float64)
    return HestonPaths(
        times=times,
        log_prices=log_prices,
        variances=variances,
        integrated_variance=integrated,
        params=params,
    )
