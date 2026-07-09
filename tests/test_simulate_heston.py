"""Heston self-consistency against the analytic CIR moments."""

from __future__ import annotations

import numpy as np
import pytest

from fxvrp.simulate import HestonParams, simulate_heston
from fxvrp.simulate.heston import expected_integrated_variance

PARAMS = HestonParams(kappa=3.0, theta=0.02, xi=0.30, rho=-0.6, v0=0.03)
S0, MU, HORIZON = 1.10, 0.0, 0.5


def test_recorded_variance_is_nonnegative(rng: np.random.Generator) -> None:
    paths = simulate_heston(
        s0=S0, mu=MU, params=PARAMS, horizon=HORIZON, n_steps=500, n_paths=200, rng=rng
    )
    assert np.all(paths.variances >= 0.0)
    assert np.all(paths.integrated_variance >= 0.0)


def test_mean_integrated_variance_matches_cir_formula(rng: np.random.Generator) -> None:
    n_paths = 20_000
    paths = simulate_heston(
        s0=S0, mu=MU, params=PARAMS, horizon=HORIZON, n_steps=1_000, n_paths=n_paths, rng=rng
    )
    theory = expected_integrated_variance(PARAMS, HORIZON)
    mc = paths.integrated_variance
    se = mc.std() / np.sqrt(n_paths)
    assert abs(mc.mean() - theory) < 4 * se + 1e-5  # MC error + O(dt) scheme bias


def test_long_run_variance_reverts_to_theta(rng: np.random.Generator) -> None:
    long_horizon = 20.0
    paths = simulate_heston(
        s0=S0,
        mu=MU,
        params=PARAMS,
        horizon=long_horizon,
        n_steps=8_000,
        n_paths=4_000,
        rng=rng,
    )
    terminal_v = paths.variances[:, -1]
    # stationary CIR mean is theta; se uses the stationary std xi*sqrt(theta/(2 kappa))
    stat_std = PARAMS.xi * np.sqrt(PARAMS.theta / (2.0 * PARAMS.kappa))
    se = stat_std / np.sqrt(terminal_v.size)
    assert abs(terminal_v.mean() - PARAMS.theta) < 5 * se + 1e-4


def test_squared_returns_track_path_integrated_variance(rng: np.random.Generator) -> None:
    """Σ r² per path ≈ that path's ∫v dt: the RV ground-truth contract for Phase 2."""
    paths = simulate_heston(
        s0=S0, mu=MU, params=PARAMS, horizon=HORIZON, n_steps=50_000, n_paths=16, rng=rng
    )
    rv = (paths.log_returns() ** 2).sum(axis=1)
    assert np.corrcoef(rv, paths.integrated_variance)[0, 1] > 0.999
    assert np.allclose(rv, paths.integrated_variance, rtol=0.10)


def test_feller_flag() -> None:
    assert HestonParams(kappa=3.0, theta=0.02, xi=0.30, rho=0.0, v0=0.02).feller_satisfied
    assert not HestonParams(kappa=1.0, theta=0.01, xi=0.30, rho=0.0, v0=0.01).feller_satisfied


@pytest.mark.parametrize(
    "params",
    [
        HestonParams(kappa=-1.0, theta=0.02, xi=0.3, rho=0.0, v0=0.02),
        HestonParams(kappa=1.0, theta=0.02, xi=0.3, rho=1.5, v0=0.02),
        HestonParams(kappa=1.0, theta=0.02, xi=0.3, rho=0.0, v0=-0.1),
    ],
)
def test_invalid_params_raise(rng: np.random.Generator, params: HestonParams) -> None:
    with pytest.raises(ValueError):
        simulate_heston(
            s0=S0, mu=MU, params=params, horizon=HORIZON, n_steps=10, n_paths=2, rng=rng
        )
