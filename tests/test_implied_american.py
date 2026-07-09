"""CRR binomial: convergence to BS, early-exercise structure, de-Americanization."""

from __future__ import annotations

import pytest

from fxvrp.implied import bs_price, crr_price, de_americanize, implied_vol

S, K, T, SIGMA = 105.0, 100.0, 30.0 / 365.0, 0.09
R, Q = 0.036, 0.022
STEPS = 500


def test_european_binomial_converges_to_black_scholes() -> None:
    for call in (True, False):
        for k in (95.0, 105.0, 115.0):
            bs = bs_price(s=S, k=k, t=T, r=R, q=Q, sigma=SIGMA, call=call)
            crr = crr_price(
                s=S, k=k, t=T, r=R, q=Q, sigma=SIGMA, n_steps=STEPS, call=call, american=False
            )
            assert crr == pytest.approx(bs, rel=2e-3, abs=2e-4)


def test_american_dominates_european_and_intrinsic() -> None:
    for call in (True, False):
        for k in (90.0, 100.0, 110.0):
            eu = crr_price(
                s=S, k=k, t=T, r=R, q=Q, sigma=SIGMA, n_steps=STEPS, call=call, american=False
            )
            am = crr_price(
                s=S, k=k, t=T, r=R, q=Q, sigma=SIGMA, n_steps=STEPS, call=call, american=True
            )
            intrinsic = max(S - k, 0.0) if call else max(k - S, 0.0)
            assert am >= eu - 1e-12
            assert am >= intrinsic - 1e-12


def test_american_call_without_yield_equals_european() -> None:
    """Merton (1973): early exercise of a call is never optimal when q = 0."""
    eu = crr_price(s=S, k=K, t=T, r=R, q=0.0, sigma=SIGMA, n_steps=STEPS, call=True, american=False)
    am = crr_price(s=S, k=K, t=T, r=R, q=0.0, sigma=SIGMA, n_steps=STEPS, call=True, american=True)
    assert am == pytest.approx(eu, abs=1e-10)


def test_early_exercise_premium_sits_where_phase0_measured_it() -> None:
    """r > q puts the premium on ITM puts; the ATM magnitude matches Phase 0 (~0.8%)."""
    eu = crr_price(
        s=100.0, k=100.0, t=T, r=R, q=Q, sigma=SIGMA, n_steps=STEPS, call=False, american=False
    )
    am = crr_price(
        s=100.0, k=100.0, t=T, r=R, q=Q, sigma=SIGMA, n_steps=STEPS, call=False, american=True
    )
    premium = (am - eu) / eu
    assert 0.004 < premium < 0.012
    # and the call side carries none
    eu_c = crr_price(
        s=100.0, k=100.0, t=T, r=R, q=Q, sigma=SIGMA, n_steps=STEPS, call=True, american=False
    )
    am_c = crr_price(
        s=100.0, k=100.0, t=T, r=R, q=Q, sigma=SIGMA, n_steps=STEPS, call=True, american=True
    )
    assert (am_c - eu_c) / eu_c < 1e-4


def test_de_americanize_round_trip() -> None:
    for call in (True, False):
        for k in (95.0, 100.0, 105.0):
            price = crr_price(
                s=S, k=k, t=T, r=R, q=Q, sigma=SIGMA, n_steps=STEPS, call=call, american=True
            )
            recovered = de_americanize(
                price=price, s=S, k=k, t=T, r=R, q=Q, call=call, n_steps=STEPS
            )
            assert recovered == pytest.approx(SIGMA, rel=1e-4)


def test_de_americanize_differs_from_naive_european_inversion() -> None:
    """Inverting an American put quote with the European model overstates vol."""
    am_put = crr_price(
        s=100.0,
        k=103.0,
        t=60.0 / 365.0,
        r=0.055,
        q=0.039,
        sigma=0.06,
        n_steps=STEPS,
        call=False,
        american=True,
    )
    naive = implied_vol(
        price=am_put, s=100.0, k=103.0, t=60.0 / 365.0, r=0.055, q=0.039, call=False
    )
    proper = de_americanize(
        price=am_put, s=100.0, k=103.0, t=60.0 / 365.0, r=0.055, q=0.039, call=False, n_steps=STEPS
    )
    assert proper == pytest.approx(0.06, rel=1e-3)
    assert naive > proper * 1.02  # the premium masquerades as extra vol


def test_binomial_input_validation() -> None:
    with pytest.raises(ValueError):
        crr_price(s=S, k=K, t=T, r=R, q=Q, sigma=SIGMA, n_steps=0, call=True, american=True)
    with pytest.raises(ValueError):
        de_americanize(price=1.0, s=S, k=K, t=0.0, r=R, q=Q, call=True, n_steps=STEPS)
    with pytest.raises(ValueError, match="below American lower bound"):
        de_americanize(price=0.0001, s=120.0, k=100.0, t=T, r=R, q=Q, call=True, n_steps=STEPS)
