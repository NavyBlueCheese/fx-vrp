"""Noise models must show exactly the signatures the estimators will be built to defeat."""

from __future__ import annotations

import numpy as np
import pytest

from fxvrp.simulate import add_bid_ask_bounce, add_gaussian_noise, simulate_gbm

S0, SIGMA, HORIZON = 1.10, 0.09, 1.0 / 12.0


def test_gaussian_noise_inflates_return_variance_by_2_eta_sq(
    rng: np.random.Generator,
) -> None:
    """Var(observed return) = σ²dt + 2η² — ZMA (2005) eq. (2)'s implication."""
    n_steps = 500_000
    eta = 5e-5
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=n_steps, n_paths=1, rng=rng
    )
    noisy = add_gaussian_noise(paths.log_prices, eta, rng)
    observed_var = np.diff(noisy, axis=1).var()
    dt = HORIZON / n_steps
    theory = SIGMA**2 * dt + 2 * eta**2
    assert observed_var == pytest.approx(theory, rel=0.02)


def test_gaussian_noise_makes_naive_rv_diverge(rng: np.random.Generator) -> None:
    """RV grows without bound in the sampling frequency under noise — the failure
    mode the volatility signature plot displays (demonstrated, not assumed)."""
    eta = 1e-4  # noise floor: E[RV_fine] ≈ σ²T + 2 n η², here ≈ 13× the true IV
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=400_000, n_paths=1, rng=rng
    )
    noisy = add_gaussian_noise(paths.log_prices, eta, rng)[0]
    true_iv = paths.integrated_variance

    def rv_at_stride(stride: int) -> float:
        return float((np.diff(noisy[::stride]) ** 2).sum())

    rv_fine, rv_mid, rv_coarse = rv_at_stride(1), rv_at_stride(100), rv_at_stride(1_000)
    assert rv_fine > 10 * true_iv  # hopelessly biased at tick frequency
    assert rv_fine > rv_mid > rv_coarse  # bias grows monotonically in frequency
    assert rv_coarse == pytest.approx(true_iv, rel=0.5)  # tame at coarse sampling


def test_bounce_creates_negative_first_order_autocovariance(
    rng: np.random.Generator,
) -> None:
    """Roll (1984): Cov(r_i, r_{i-1}) ≈ −c² under bid-ask bounce."""
    c = 8e-5
    n_steps = 500_000
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=n_steps, n_paths=1, rng=rng
    )
    bounced = add_bid_ask_bounce(paths.log_prices, c, rng)[0]
    returns = np.diff(bounced)
    autocov = np.cov(returns[1:], returns[:-1])[0, 1]
    assert autocov == pytest.approx(-(c**2), rel=0.10)


def test_noise_is_zero_when_scale_is_zero(rng: np.random.Generator) -> None:
    log_prices = np.zeros((2, 10))
    assert np.array_equal(add_gaussian_noise(log_prices, 0.0, rng), log_prices)


def test_negative_scales_raise(rng: np.random.Generator) -> None:
    log_prices = np.zeros((1, 4))
    with pytest.raises(ValueError):
        add_gaussian_noise(log_prices, -1e-4, rng)
    with pytest.raises(ValueError):
        add_bid_ask_bounce(log_prices, -1e-4, rng)
