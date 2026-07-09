"""Microstructure contamination of efficient log prices.

Two canonical noise models:

1. Additive i.i.d. Gaussian noise — the setting of Zhang, Mykland & Aït-Sahalia
   (2005, JASA 100(472)), eq. (1)-(2): observed Y_i = X_i + u_i with
   u_i ~ N(0, eta^2) independent of X. Observed per-step returns then have variance
   Var(ΔX) + 2 eta^2, which is what makes naive RV diverge as sampling accelerates.

2. Bid-ask bounce — Roll (1984, J. Finance 39(4)): Y_i = X_i + c q_i with
   q_i = ±1 i.i.d. (trade direction) and c the effective half-spread. Roll's model
   implies first-order return autocovariance ≈ −c².

Both take and return *log prices* and are pure functions of (input, rng).
"""

from __future__ import annotations

import numpy as np

from fxvrp._types import FloatArray


def add_gaussian_noise(
    log_prices: FloatArray,
    noise_std: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Contaminate log prices with i.i.d. N(0, noise_std^2) observation noise."""
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    noise = rng.normal(0.0, noise_std, size=log_prices.shape)
    return np.asarray(log_prices + noise, dtype=np.float64)


def add_bid_ask_bounce(
    log_prices: FloatArray,
    half_spread: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Contaminate log prices with Roll (1984) bid-ask bounce of half-spread c."""
    if half_spread < 0.0:
        raise ValueError(f"half_spread must be non-negative, got {half_spread}")
    directions = rng.choice(np.array([-1.0, 1.0]), size=log_prices.shape)
    return np.asarray(log_prices + half_spread * directions, dtype=np.float64)
