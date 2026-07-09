"""Black-Scholes pricing, greeks vs finite differences, and the IV round-trip."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fxvrp.implied import bs_greeks, bs_price, implied_vol

S, K, T, R, Q, SIGMA = 105.0, 100.0, 30.0 / 365.0, 0.036, 0.022, 0.09


def test_put_call_parity_is_exact() -> None:
    call = bs_price(s=S, k=K, t=T, r=R, q=Q, sigma=SIGMA, call=True)
    put = bs_price(s=S, k=K, t=T, r=R, q=Q, sigma=SIGMA, call=False)
    forward_leg = S * np.exp(-Q * T) - K * np.exp(-R * T)
    assert call - put == pytest.approx(forward_leg, abs=1e-12)


def test_price_is_monotone_in_vol_and_bounded() -> None:
    prices = [bs_price(s=S, k=K, t=T, r=R, q=Q, sigma=v, call=True) for v in (0.05, 0.10, 0.30)]
    assert prices[0] < prices[1] < prices[2]
    assert prices[0] > bs_price(s=S, k=K, t=T, r=R, q=Q, sigma=0.0, call=True)
    assert prices[2] < S * np.exp(-Q * T)


def test_degenerate_cases() -> None:
    assert bs_price(s=S, k=K, t=0.0, r=R, q=Q, sigma=SIGMA, call=True) == S - K
    assert bs_price(s=90.0, k=K, t=0.0, r=R, q=Q, sigma=SIGMA, call=True) == 0.0
    intrinsic = S * np.exp(-Q * T) - K * np.exp(-R * T)
    assert bs_price(s=S, k=K, t=T, r=R, q=Q, sigma=0.0, call=True) == pytest.approx(intrinsic)


def test_greeks_match_finite_differences() -> None:
    eps_s, eps_v, eps_t, eps_r = 1e-4, 1e-6, 1e-7, 1e-7
    for call in (True, False):
        greeks = bs_greeks(s=S, k=K, t=T, r=R, q=Q, sigma=SIGMA, call=call)

        def price(s: float = S, sigma: float = SIGMA, t: float = T, r: float = R) -> float:
            return bs_price(s=s, k=K, t=t, r=r, q=Q, sigma=sigma, call=call)  # noqa: B023

        delta_fd = (price(s=S + eps_s) - price(s=S - eps_s)) / (2 * eps_s)
        gamma_fd = (price(s=S + eps_s) - 2 * price() + price(s=S - eps_s)) / eps_s**2
        vega_fd = (price(sigma=SIGMA + eps_v) - price(sigma=SIGMA - eps_v)) / (2 * eps_v)
        theta_fd = -(price(t=T + eps_t) - price(t=T - eps_t)) / (2 * eps_t)
        rho_fd = (price(r=R + eps_r) - price(r=R - eps_r)) / (2 * eps_r)

        assert greeks.delta == pytest.approx(delta_fd, rel=1e-5)
        assert greeks.gamma == pytest.approx(gamma_fd, rel=1e-3)
        assert greeks.vega == pytest.approx(vega_fd, rel=1e-5)
        assert greeks.theta == pytest.approx(theta_fd, rel=1e-4)
        assert greeks.rho == pytest.approx(rho_fd, rel=1e-5)


@given(
    sigma=st.floats(0.001, 5.0),
    moneyness=st.floats(0.7, 1.4),
    call=st.booleans(),
)
@settings(max_examples=200, deadline=None)
def test_iv_round_trip(sigma: float, moneyness: float, call: bool) -> None:
    """IV(price(σ)) == σ across the brief's required range σ ∈ [0.001, 5.0]."""
    k = S * moneyness
    price = bs_price(s=S, k=k, t=T, r=R, q=Q, sigma=sigma, call=call)
    lower = bs_price(s=S, k=k, t=T, r=R, q=Q, sigma=0.0, call=call)
    if price - lower < 1e-9:  # vol numerically invisible for this deep quote
        return
    recovered = implied_vol(price=price, s=S, k=k, t=T, r=R, q=Q, call=call)
    assert recovered == pytest.approx(sigma, rel=1e-5, abs=1e-6)


def test_iv_rejects_arbitrage_violations() -> None:
    upper = S * np.exp(-Q * T)
    with pytest.raises(ValueError, match="outside no-arbitrage"):
        implied_vol(price=upper * 1.01, s=S, k=K, t=T, r=R, q=Q, call=True)
    with pytest.raises(ValueError, match="outside no-arbitrage"):
        implied_vol(price=-0.01, s=S, k=K, t=T, r=R, q=Q, call=True)
    with pytest.raises(ValueError, match="t=0"):
        implied_vol(price=1.0, s=S, k=K, t=0.0, r=R, q=Q, call=True)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        bs_price(s=-1.0, k=K, t=T, r=R, q=Q, sigma=SIGMA, call=True)
    with pytest.raises(ValueError):
        bs_price(s=S, k=K, t=-0.1, r=R, q=Q, sigma=SIGMA, call=True)
    with pytest.raises(ValueError):
        bs_greeks(s=S, k=K, t=0.0, r=R, q=Q, sigma=SIGMA, call=True)
