"""Ground-truth table (§3), jump rows:

Merton jump-diffusion → BPV recovers the continuous part only
Merton jump-diffusion → RV − BPV → the jump part
GBM (no jumps)        → BNS test has correct size under H₀
Merton (with jumps)   → BNS test has power under H₁
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from fxvrp.realized import (
    bipower_variation,
    bns_test_statistic,
    jump_variation,
    tripower_quarticity,
)
from fxvrp.simulate import MertonParams, simulate_gbm, simulate_merton

S0, SIGMA = 1.10, 0.09
HORIZON = 1.0 / 12.0
CONT_IV = SIGMA**2 * HORIZON
JUMPY = MertonParams(sigma=SIGMA, jump_intensity=60.0, jump_mean=0.0, jump_std=0.01)


def test_bpv_recovers_continuous_part_under_jumps(rng: np.random.Generator) -> None:
    paths = simulate_merton(
        s0=S0, mu=0.0, params=JUMPY, horizon=HORIZON, n_steps=20_000, n_paths=64, rng=rng
    )
    bpv = np.asarray(bipower_variation(paths.log_returns()))
    # BPV estimates sigma^2 T regardless of the jumps the paths actually carry
    assert bpv.mean() == pytest.approx(CONT_IV, rel=0.03)
    # while raw RV is inflated by exactly the jump variation
    rv = (paths.log_returns() ** 2).sum(axis=1)
    assert rv.mean() > bpv.mean()
    assert (rv - bpv).mean() == pytest.approx(paths.jump_variation.mean(), rel=0.15)


def test_rv_minus_bpv_recovers_path_jump_variation(rng: np.random.Generator) -> None:
    """Per path: JV_hat tracks the sum of that path's actual squared jumps."""
    params = MertonParams(sigma=SIGMA, jump_intensity=40.0, jump_mean=-0.02, jump_std=0.01)
    paths = simulate_merton(
        s0=S0, mu=0.0, params=params, horizon=HORIZON, n_steps=50_000, n_paths=48, rng=rng
    )
    jv_hat = np.asarray(jump_variation(paths.log_returns()))
    # judge accuracy on paths whose jumps are visible above discretisation noise
    jumped = paths.jump_variation > 1e-5
    assert jumped.sum() >= 30  # intensity high enough that most paths jumped
    corr = np.corrcoef(jv_hat[jumped], paths.jump_variation[jumped])[0, 1]
    assert corr > 0.99
    assert np.allclose(jv_hat[jumped], paths.jump_variation[jumped], rtol=0.35, atol=2e-6)


def test_bpv_unbiased_without_jumps(rng: np.random.Generator) -> None:
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=5_000, n_paths=128, rng=rng
    )
    bpv = np.asarray(bipower_variation(paths.log_returns()))
    assert bpv.mean() == pytest.approx(CONT_IV, rel=0.02)


def test_tripower_quarticity_normalisation(rng: np.random.Generator) -> None:
    """TQ/BPV² → 1 under constant volatility — the scaling the z-test relies on."""
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=HORIZON, n_steps=20_000, n_paths=64, rng=rng
    )
    returns = paths.log_returns()
    tq = np.asarray(tripower_quarticity(returns))
    bpv = np.asarray(bipower_variation(returns))
    assert (tq / bpv**2).mean() == pytest.approx(1.0, rel=0.02)


@pytest.mark.slow
def test_bns_size_under_null(rng: np.random.Generator) -> None:
    """Rejection rate on jump-free days ≈ nominal α (binomial tolerance)."""
    alpha = 0.05
    n_days = 800
    paths = simulate_gbm(
        s0=S0, mu=0.0, sigma=SIGMA, horizon=1.0 / 252.0, n_steps=288, n_paths=n_days, rng=rng
    )
    critical = stats.norm.ppf(1.0 - alpha)
    z = np.array([bns_test_statistic(r) for r in paths.log_returns()])
    rejection_rate = float(np.mean(z > critical))
    # 3 binomial standard errors around alpha, plus finite-sample slack
    tol = 3.0 * np.sqrt(alpha * (1 - alpha) / n_days) + 0.02
    assert abs(rejection_rate - alpha) < tol


@pytest.mark.slow
def test_bns_power_under_alternative(rng: np.random.Generator) -> None:
    """Days carrying a visible jump should be flagged most of the time."""
    alpha = 0.05
    params = MertonParams(sigma=SIGMA, jump_intensity=252.0, jump_mean=0.0, jump_std=0.004)
    paths = simulate_merton(
        s0=S0, mu=0.0, params=params, horizon=1.0 / 252.0, n_steps=288, n_paths=800, rng=rng
    )
    z = np.array([bns_test_statistic(r) for r in paths.log_returns()])
    jumped = paths.jump_variation > 0
    assert jumped.sum() > 300
    critical = stats.norm.ppf(1.0 - alpha)
    power = float(np.mean(z[jumped] > critical))
    assert power > 0.60


def test_jump_statistic_input_validation() -> None:
    with pytest.raises(ValueError):
        bns_test_statistic(np.zeros((2, 5)))
    with pytest.raises(ValueError):
        bipower_variation(np.zeros(1))
    with pytest.raises(ValueError):
        tripower_quarticity(np.zeros(2))
    assert bns_test_statistic(np.zeros(10)) == 0.0  # degenerate day, no signal
