"""Merton (1976) jump diffusion with per-jump bookkeeping.

Model (log form): over a step of length dt,
    Δ log S = (mu - sigma^2/2 - lambda kbar) dt + sigma sqrt(dt) Z + Σ_{j=1}^{N} Y_j,
    N ~ Poisson(lambda dt),  Y_j ~ N(mu_j, sigma_j^2) i.i.d.,
    kbar = E[e^Y] - 1 = exp(mu_j + sigma_j^2/2) - 1.

The drift is compensated so that E[S_T] = S_0 e^{mu T} exactly (Merton 1976,
*J. Financial Economics* 3, eq. (2)-(3) with his alpha = mu here).

Ground truth recorded per path:
  - continuous quadratic variation  sigma^2 T (constant by construction),
  - jump variation                  Σ_j Y_j^2 over *individual* jumps,
so QV(log S) = sigma^2 T + Σ Y_j^2. Bipower variation must recover the first term
and RV − BPV the second (the Phase 2 acceptance tests).

Individual jump sizes are drawn explicitly (not aggregated normally) because the
jump variation needs Σ Y_j^2, which the sum of a step's jumps does not determine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fxvrp._types import FloatArray, IntArray


@dataclass(frozen=True)
class MertonParams:
    sigma: float  # diffusive volatility
    jump_intensity: float  # lambda, jumps per year
    jump_mean: float  # mu_j, mean log-jump size
    jump_std: float  # sigma_j

    def validate(self) -> None:
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if self.jump_intensity < 0.0:
            raise ValueError(f"jump_intensity must be non-negative, got {self.jump_intensity}")
        if self.jump_std < 0.0:
            raise ValueError(f"jump_std must be non-negative, got {self.jump_std}")

    @property
    def kbar(self) -> float:
        """Expected relative jump size E[e^Y] - 1."""
        return float(np.exp(self.jump_mean + self.jump_std**2 / 2.0) - 1.0)


@dataclass(frozen=True)
class MertonPaths:
    times: FloatArray  # (n_steps + 1,)
    log_prices: FloatArray  # (n_paths, n_steps + 1)
    jump_increments: FloatArray  # (n_paths, n_steps), summed jump sizes per step
    jump_counts: IntArray  # (n_paths, n_steps)
    jump_variation: FloatArray  # (n_paths,), Σ Y_j^2 over individual jumps
    mu: float
    params: MertonParams

    @property
    def prices(self) -> FloatArray:
        return np.exp(self.log_prices)

    @property
    def horizon(self) -> float:
        return float(self.times[-1])

    @property
    def continuous_variance(self) -> float:
        """Quadratic variation of the diffusive component: sigma^2 T."""
        return self.params.sigma**2 * self.horizon

    def total_quadratic_variation(self) -> FloatArray:
        """Per-path QV of log S: sigma^2 T + Σ Y_j^2."""
        return self.continuous_variance + self.jump_variation

    def log_returns(self) -> FloatArray:
        return np.diff(self.log_prices, axis=1)


def simulate_merton(
    *,
    s0: float,
    mu: float,
    params: MertonParams,
    horizon: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> MertonPaths:
    """Simulate Merton jump-diffusion paths, recording every individual jump."""
    params.validate()
    if s0 <= 0.0:
        raise ValueError(f"s0 must be positive, got {s0}")
    if horizon <= 0.0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be at least 1")

    dt = horizon / n_steps
    drift = (mu - params.sigma**2 / 2.0 - params.jump_intensity * params.kbar) * dt
    diffusion = params.sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))

    counts = rng.poisson(params.jump_intensity * dt, size=(n_paths, n_steps))
    jump_increments = np.zeros((n_paths, n_steps), dtype=np.float64)
    jump_variation = np.zeros(n_paths, dtype=np.float64)

    # jumps are rare at fine dt; draw sizes only where counts > 0, jump by jump,
    # because the jump variation needs each Y_j^2 individually
    path_idx, step_idx = np.nonzero(counts)
    for path, step in zip(path_idx.tolist(), step_idx.tolist(), strict=True):
        sizes = rng.normal(params.jump_mean, params.jump_std, size=int(counts[path, step]))
        jump_increments[path, step] = float(np.sum(sizes))
        jump_variation[path] += float(np.sum(sizes**2))

    increments = drift + diffusion + jump_increments
    log_prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_prices[:, 0] = np.log(s0)
    np.cumsum(increments, axis=1, out=log_prices[:, 1:])
    log_prices[:, 1:] += np.log(s0)

    times = np.linspace(0.0, horizon, n_steps + 1, dtype=np.float64)
    return MertonPaths(
        times=times,
        log_prices=log_prices,
        jump_increments=jump_increments,
        jump_counts=counts.astype(np.int64),
        jump_variation=jump_variation,
        mu=mu,
        params=params,
    )
