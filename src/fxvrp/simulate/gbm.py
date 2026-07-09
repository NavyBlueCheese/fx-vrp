"""Geometric Brownian motion, sampled exactly.

Model: dS_t = mu S_t dt + sigma S_t dW_t, so that
log S_T ~ N(log S_0 + (mu - sigma^2/2) T, sigma^2 T) and the quadratic variation of
log S over [0, T] is sigma^2 T. The log-normal step is exact at any step size
(Glasserman, 2004, *Monte Carlo Methods in Financial Engineering*, §3.2.1), so
discretization bias in downstream estimator tests is zero by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fxvrp._types import FloatArray


@dataclass(frozen=True)
class GBMPaths:
    """Simulated GBM paths and the ground truth they carry."""

    times: FloatArray  # shape (n_steps + 1,), in years
    log_prices: FloatArray  # shape (n_paths, n_steps + 1)
    mu: float
    sigma: float

    @property
    def prices(self) -> FloatArray:
        return np.exp(self.log_prices)

    @property
    def horizon(self) -> float:
        return float(self.times[-1])

    @property
    def integrated_variance(self) -> float:
        """True quadratic variation of log S over the horizon: sigma^2 T."""
        return self.sigma**2 * self.horizon

    def log_returns(self) -> FloatArray:
        """Per-step log returns, shape (n_paths, n_steps)."""
        return np.diff(self.log_prices, axis=1)


def simulate_gbm(
    *,
    s0: float,
    mu: float,
    sigma: float,
    horizon: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> GBMPaths:
    """Simulate GBM log-price paths on an equispaced grid over ``horizon`` years."""
    if s0 <= 0.0:
        raise ValueError(f"s0 must be positive, got {s0}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if horizon <= 0.0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be at least 1")

    dt = horizon / n_steps
    shocks = rng.standard_normal((n_paths, n_steps))
    increments = (mu - sigma**2 / 2.0) * dt + sigma * np.sqrt(dt) * shocks
    log_prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_prices[:, 0] = np.log(s0)
    np.cumsum(increments, axis=1, out=log_prices[:, 1:])
    log_prices[:, 1:] += np.log(s0)

    times = np.linspace(0.0, horizon, n_steps + 1, dtype=np.float64)
    return GBMPaths(times=times, log_prices=log_prices, mu=mu, sigma=sigma)
