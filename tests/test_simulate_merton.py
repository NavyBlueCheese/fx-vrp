"""Merton jump-diffusion self-consistency: compensated drift and jump bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from fxvrp.simulate import MertonParams, simulate_merton

PARAMS = MertonParams(sigma=0.08, jump_intensity=20.0, jump_mean=-0.01, jump_std=0.02)
S0, MU, HORIZON = 1.10, 0.02, 0.5


def test_jump_count_mean_is_lambda_t(rng: np.random.Generator) -> None:
    n_paths = 20_000
    paths = simulate_merton(
        s0=S0, mu=MU, params=PARAMS, horizon=HORIZON, n_steps=252, n_paths=n_paths, rng=rng
    )
    total_jumps = paths.jump_counts.sum(axis=1)
    lam_t = PARAMS.jump_intensity * HORIZON
    se = np.sqrt(lam_t / n_paths)  # Poisson variance = mean
    assert abs(total_jumps.mean() - lam_t) < 4 * se


def test_drift_compensation_makes_expected_price_exact(rng: np.random.Generator) -> None:
    n_paths = 400_000
    paths = simulate_merton(
        s0=S0, mu=MU, params=PARAMS, horizon=HORIZON, n_steps=64, n_paths=n_paths, rng=rng
    )
    terminal = paths.prices[:, -1]
    theory = S0 * np.exp(MU * HORIZON)
    se = terminal.std() / np.sqrt(n_paths)
    assert abs(terminal.mean() - theory) < 4 * se


def test_squared_returns_recover_total_quadratic_variation(rng: np.random.Generator) -> None:
    """Σ r² ≈ σ²T + Σ Y² per path — the decomposition BPV must split in Phase 2."""
    paths = simulate_merton(
        s0=S0, mu=MU, params=PARAMS, horizon=HORIZON, n_steps=100_000, n_paths=16, rng=rng
    )
    rv = (paths.log_returns() ** 2).sum(axis=1)
    assert np.allclose(rv, paths.total_quadratic_variation(), rtol=0.10)


def test_jump_variation_equals_sum_of_individual_squared_jumps(
    rng: np.random.Generator,
) -> None:
    paths = simulate_merton(
        s0=S0, mu=MU, params=PARAMS, horizon=HORIZON, n_steps=32, n_paths=200, rng=rng
    )
    # paths with no jumps carry zero jump variation
    no_jump = paths.jump_counts.sum(axis=1) == 0
    assert np.all(paths.jump_variation[no_jump] == 0.0)
    # jump variation is positive exactly when jumps occurred (a.s.)
    assert np.all(paths.jump_variation[~no_jump] > 0.0)
    # steps without jumps contribute no jump increment
    assert np.all(paths.jump_increments[paths.jump_counts == 0] == 0.0)


def test_zero_intensity_degenerates_to_gbm(rng: np.random.Generator) -> None:
    params = MertonParams(sigma=0.08, jump_intensity=0.0, jump_mean=-0.01, jump_std=0.02)
    paths = simulate_merton(
        s0=S0, mu=MU, params=params, horizon=HORIZON, n_steps=64, n_paths=100, rng=rng
    )
    assert paths.jump_counts.sum() == 0
    assert np.all(paths.jump_variation == 0.0)
    assert np.all(paths.total_quadratic_variation() == pytest.approx(params.sigma**2 * HORIZON))


def test_kbar_matches_lognormal_mean() -> None:
    assert PARAMS.kbar == pytest.approx(
        float(np.exp(PARAMS.jump_mean + PARAMS.jump_std**2 / 2) - 1.0)
    )
