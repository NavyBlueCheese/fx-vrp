"""Ground-truth table (§3 of the project brief), realised-variance rows:

GBM, known σ            → RV → σ²T as sampling refines
Heston, path-recorded   → RV recovers the *path's* ∫v dt
GBM + additive noise    → naive RV biased upward, diverging as Δ→0
GBM + additive noise    → TSRV and realised kernel recover σ²T
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fxvrp.realized import (
    realized_kernel,
    realized_semivariance,
    realized_variance,
    subsampled_rv,
    tsrv,
)
from fxvrp.realized.estimators import kernel_bandwidth
from fxvrp.simulate import HestonParams, add_gaussian_noise, simulate_gbm, simulate_heston

S0, SIGMA, HORIZON = 1.10, 0.09, 1.0 / 12.0
TRUE_IV = SIGMA**2 * HORIZON


def test_rv_converges_to_sigma_sq_t_under_gbm(rng: np.random.Generator) -> None:
    """Bias shrinks as the grid refines; at 5-min-equivalent it is already small."""
    n_paths = 200
    errors = []
    for n_steps in (72, 288, 8_640):  # 8h, 5-min-equivalent, 5-sec-equivalent grids
        paths = simulate_gbm(
            s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=n_steps, n_paths=n_paths, rng=rng
        )
        rv = np.asarray(realized_variance(paths.log_returns()))
        errors.append(abs(rv.mean() - TRUE_IV) / TRUE_IV)
    assert errors[-1] < 0.01  # unbiased to 1% at fine sampling
    # RV of n iid squared Gaussians has relative std sqrt(2/n); check consistency
    assert np.std(rv) / TRUE_IV == pytest.approx(np.sqrt(2.0 / 8_640), rel=0.25)


def test_rv_recovers_path_integrated_variance_under_heston(rng: np.random.Generator) -> None:
    """RV must track each path's own ∫v dt, not just the ensemble mean."""
    params = HestonParams(kappa=3.0, theta=0.02, xi=0.30, rho=-0.6, v0=0.03)
    paths = simulate_heston(
        s0=S0, mu=0.0, params=params, horizon=0.25, n_steps=50_000, n_paths=24, rng=rng
    )
    rv = np.asarray(realized_variance(paths.log_returns()))
    assert np.corrcoef(rv, paths.integrated_variance)[0, 1] > 0.999
    assert np.allclose(rv, paths.integrated_variance, rtol=0.10)


def test_naive_rv_diverges_under_noise_but_tsrv_recovers(rng: np.random.Generator) -> None:
    """The central microstructure result, demonstrated through the estimator API.

    TSRV converges at the slow n^{-1/6} rate (~15% relative std per path at
    n = 100k), so accuracy is asserted on a cross-path average.
    """
    eta = 1e-4
    n_steps, n_paths = 100_000, 8
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=n_steps, n_paths=n_paths, rng=rng
    )
    noisy = add_gaussian_noise(paths.log_prices, eta, rng)

    rv_noisy = np.array([float(realized_variance(np.diff(p))) for p in noisy])
    expected_bias = 2.0 * n_steps * eta**2  # ≈ 3x the true IV at these scales
    assert rv_noisy.mean() > 3 * TRUE_IV  # naive RV is destroyed
    assert rv_noisy.mean() == pytest.approx(TRUE_IV + expected_bias, rel=0.05)

    # TSRV strips the noise (K near the ZMA rate n^{2/3})
    k = int(n_steps ** (2.0 / 3.0))
    tsrv_vals = np.array([tsrv(p, k) for p in noisy])
    se = tsrv_vals.std() / np.sqrt(n_paths)
    assert abs(tsrv_vals.mean() - TRUE_IV) < 4 * se + 0.05 * TRUE_IV
    # and TSRV is unbiased on clean data too (agrees with the noise-free RV)
    tsrv_clean = np.array([tsrv(p, k) for p in paths.log_prices])
    assert tsrv_clean.mean() == pytest.approx(TRUE_IV, rel=0.15)


def test_realized_kernel_recovers_iv_under_noise(rng: np.random.Generator) -> None:
    eta = 1e-4
    n_steps = 100_000
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=n_steps, n_paths=1, rng=rng
    )
    noisy_returns = np.diff(add_gaussian_noise(paths.log_prices, eta, rng)[0])

    bandwidth = kernel_bandwidth(noisy_returns, sparse_stride=300)
    rk = realized_kernel(noisy_returns, bandwidth)
    assert rk == pytest.approx(TRUE_IV, rel=0.20)
    # and the kernel with H=0 degenerates to naive RV (hopelessly biased)
    assert realized_kernel(noisy_returns, 0) == pytest.approx(
        float(realized_variance(noisy_returns)), rel=1e-12
    )


def test_subsampled_rv_averages_offsets(rng: np.random.Generator) -> None:
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=1_000, n_paths=1, rng=rng
    )
    prices = paths.log_prices[0]
    direct = np.mean([float(realized_variance(np.diff(prices[k::10]))) for k in range(10)])
    assert subsampled_rv(prices, 10) == pytest.approx(direct)


def test_semivariance_identity_and_symmetry(rng: np.random.Generator) -> None:
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=20_000, n_paths=8, rng=rng
    )
    returns = paths.log_returns()
    rs_plus, rs_minus = realized_semivariance(returns)
    rv = np.asarray(realized_variance(returns))
    assert np.allclose(rs_plus + rs_minus, rv, rtol=1e-12)  # exact identity
    # driftless Gaussian world: each side carries half the variance
    assert np.mean(rs_plus / rv) == pytest.approx(0.5, abs=0.02)


@given(scale=st.floats(0.1, 10.0), n=st.integers(4, 200))
@settings(max_examples=40, deadline=None)
def test_rv_scale_equivariance(scale: float, n: int) -> None:
    rng = np.random.default_rng(11)
    returns = rng.standard_normal(n) * 1e-4
    assert float(realized_variance(scale * returns)) == pytest.approx(
        scale**2 * float(realized_variance(returns)), rel=1e-9
    )


def test_estimator_input_validation() -> None:
    prices = np.zeros(10)
    with pytest.raises(ValueError):
        tsrv(prices, 1)
    with pytest.raises(ValueError):
        tsrv(prices, 9)
    with pytest.raises(ValueError):
        subsampled_rv(prices, 0)
    with pytest.raises(ValueError):
        realized_kernel(np.zeros(5), 5)
