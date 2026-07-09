"""GBM self-consistency: the exact scheme must reproduce its own analytics."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fxvrp.simulate import simulate_gbm

S0, MU, SIGMA, HORIZON = 1.10, 0.03, 0.09, 0.25


def test_log_return_moments_match_theory(rng: np.random.Generator) -> None:
    n_steps, n_paths = 64, 40_000
    paths = simulate_gbm(
        s0=S0, mu=MU, sigma=SIGMA, horizon=HORIZON, n_steps=n_steps, n_paths=n_paths, rng=rng
    )
    dt = HORIZON / n_steps
    returns = paths.log_returns()

    mean_theory = (MU - SIGMA**2 / 2) * dt
    std_theory = SIGMA * np.sqrt(dt)
    n_obs = returns.size
    # sample mean of n_obs iid draws: se = sigma_step / sqrt(n_obs)
    assert abs(returns.mean() - mean_theory) < 4 * std_theory / np.sqrt(n_obs)
    assert np.isclose(returns.std(), std_theory, rtol=0.02)


def test_terminal_price_mean_is_lognormal_mean(rng: np.random.Generator) -> None:
    paths = simulate_gbm(
        s0=S0, mu=MU, sigma=SIGMA, horizon=HORIZON, n_steps=8, n_paths=200_000, rng=rng
    )
    terminal = paths.prices[:, -1]
    theory = S0 * np.exp(MU * HORIZON)
    se = terminal.std() / np.sqrt(terminal.size)
    assert abs(terminal.mean() - theory) < 4 * se


def test_integrated_variance_is_sigma_squared_t(rng: np.random.Generator) -> None:
    paths = simulate_gbm(s0=S0, mu=MU, sigma=SIGMA, horizon=HORIZON, n_steps=16, n_paths=4, rng=rng)
    assert paths.integrated_variance == pytest.approx(SIGMA**2 * HORIZON)


def test_realized_variance_converges_to_integrated_variance(rng: np.random.Generator) -> None:
    """Σ r² → σ²T as the grid refines — the property Phase 2 estimators rely on."""
    paths = simulate_gbm(
        s0=S0, mu=MU, sigma=SIGMA, horizon=HORIZON, n_steps=100_000, n_paths=32, rng=rng
    )
    rv = (paths.log_returns() ** 2).sum(axis=1)
    assert np.allclose(rv, paths.integrated_variance, rtol=0.05)


@given(
    s0=st.floats(0.1, 100.0),
    mu=st.floats(-0.5, 0.5),
    sigma=st.floats(0.0, 2.0),
    n_steps=st.integers(1, 64),
    n_paths=st.integers(1, 16),
)
@settings(max_examples=50, deadline=None)
def test_paths_are_finite_positive_and_shaped(
    s0: float, mu: float, sigma: float, n_steps: int, n_paths: int
) -> None:
    rng = np.random.default_rng(7)
    paths = simulate_gbm(
        s0=s0, mu=mu, sigma=sigma, horizon=0.5, n_steps=n_steps, n_paths=n_paths, rng=rng
    )
    assert paths.log_prices.shape == (n_paths, n_steps + 1)
    assert paths.times.shape == (n_steps + 1,)
    assert np.all(np.isfinite(paths.log_prices))
    assert np.all(paths.prices > 0.0)
    assert np.all(paths.prices[:, 0] == pytest.approx(s0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"s0": -1.0}, "s0"),
        ({"sigma": -0.1}, "sigma"),
        ({"horizon": 0.0}, "horizon"),
        ({"n_steps": 0}, "n_steps"),
    ],
)
def test_invalid_parameters_raise(
    rng: np.random.Generator, kwargs: dict[str, float], message: str
) -> None:
    base: dict[str, float | int] = {
        "s0": S0,
        "mu": MU,
        "sigma": SIGMA,
        "horizon": HORIZON,
        "n_steps": 4,
        "n_paths": 2,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        simulate_gbm(rng=rng, **base)  # type: ignore[arg-type]
