"""Ground-truth table (§3), implied rows: MFIV on a synthetic BS chain → σ²τ."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from fxvrp.config import Config
from fxvrp.implied import (
    bs_price,
    interpolate_constant_maturity,
    select_term_expiries,
    single_expiry_variance,
)
from fxvrp.implied.mfiv import MINUTES_PER_YEAR, expiry_settlement, index_level, minutes_to

S, R, Q, SIGMA = 100.0, 0.03, 0.01, 0.20


def synthetic_chain(
    t_years: float,
    strikes: np.ndarray,
    sigma: float = SIGMA,
    spread: float = 0.02,
) -> pl.DataFrame:
    rows = []
    for k in strikes:
        for call in (True, False):
            mid = bs_price(s=S, k=float(k), t=t_years, r=R, q=Q, sigma=sigma, call=call)
            rows.append(
                {
                    "strike": float(k),
                    "call_put": "C" if call else "P",
                    "bid": max(mid * (1 - spread), 0.0),
                    "ask": mid * (1 + spread) + 1e-6,
                }
            )
    return pl.DataFrame(rows)


def test_mfiv_recovers_sigma_squared_on_bs_chain(config: Config) -> None:
    t = 30.0 / 365.0
    chain = synthetic_chain(t, np.arange(40.0, 220.0, 1.0))
    result = single_expiry_variance(chain, t_years=t, r=R, cfg=config.implied)
    # log-contract value under BS is exactly σ²; the 1-point strike grid leaves
    # ~0.5% of upward discreteness (measured), so tolerate 1%
    assert result.sigma_sq == pytest.approx(SIGMA**2, rel=1e-2)
    assert result.forward == pytest.approx(S * np.exp((R - Q) * t), rel=1e-3)
    assert result.k0 <= result.forward
    assert index_level(result.sigma_sq) == pytest.approx(100.0 * SIGMA, rel=5e-3)


def test_mfiv_insensitive_to_strike_range_extension(config: Config) -> None:
    """Wings beyond ±6 sigma add nothing: the estimate is truncation-stable."""
    t = 30.0 / 365.0
    narrow = single_expiry_variance(
        synthetic_chain(t, np.arange(70.0, 140.0, 1.0)), t_years=t, r=R, cfg=config.implied
    )
    wide = single_expiry_variance(
        synthetic_chain(t, np.arange(30.0, 300.0, 1.0)), t_years=t, r=R, cfg=config.implied
    )
    assert narrow.sigma_sq == pytest.approx(wide.sigma_sq, rel=2e-2)


def test_zero_bid_truncation_rule(config: Config) -> None:
    """One zero bid is skipped; two consecutive zero bids end the wing."""
    t = 30.0 / 365.0
    chain = synthetic_chain(t, np.arange(80.0, 125.0, 1.0))

    # kill the put bids at 84 (isolated) and at 82+81 (consecutive pair)
    def zero_bid(frame: pl.DataFrame, strike: float) -> pl.DataFrame:
        return frame.with_columns(
            pl.when((pl.col("strike") == strike) & (pl.col("call_put") == "P"))
            .then(0.0)
            .otherwise(pl.col("bid"))
            .alias("bid")
        )

    for s in (84.0, 82.0, 81.0):
        chain = zero_bid(chain, s)
    result = single_expiry_variance(chain, t_years=t, r=R, cfg=config.implied)
    full = single_expiry_variance(
        synthetic_chain(t, np.arange(80.0, 125.0, 1.0)), t_years=t, r=R, cfg=config.implied
    )
    # strikes 80, 81, 82 dropped from the put wing (83 survives; 84 skipped singly)
    assert result.n_options == full.n_options - 4  # 84 skipped + 82,81 stop + 80 unreached
    # an interior hole widens neighbouring dK and nearly re-fills the lost mass,
    # so the estimate moves only marginally (robustness, not strict monotonicity)
    assert result.sigma_sq == pytest.approx(full.sigma_sq, rel=5e-3)


def test_interpolation_is_exact_for_flat_term_structure() -> None:
    var = SIGMA**2
    n25 = 25 * 24 * 60.0
    n35 = 35 * 24 * 60.0
    interpolated = interpolate_constant_maturity(var, n25, var, n35, target_days=30)
    assert interpolated == pytest.approx(var, rel=1e-12)


def test_interpolation_weights_are_linear_in_total_variance() -> None:
    n20, n40 = 20 * 24 * 60.0, 40 * 24 * 60.0
    v1, v2 = 0.04, 0.09
    out = interpolate_constant_maturity(v1, n20, v2, n40, target_days=30)
    t1, t2 = n20 / MINUTES_PER_YEAR, n40 / MINUTES_PER_YEAR
    total = t1 * v1 * 0.5 + t2 * v2 * 0.5  # target midway: equal minute weights
    assert out == pytest.approx(total * MINUTES_PER_YEAR / (30 * 24 * 60.0))
    with pytest.raises(ValueError):
        interpolate_constant_maturity(v1, n40, v2, n20, target_days=30)


def test_select_term_expiries_prefers_straddling_pair(config: Config) -> None:
    day = 24 * 60.0
    table = {
        date(2026, 7, 17): 8 * day,
        date(2026, 8, 7): 29 * day,
        date(2026, 8, 21): 43 * day,
        date(2026, 9, 18): 71 * day,
    }
    near, nxt = select_term_expiries(table, config.implied)
    assert (near, nxt) == (date(2026, 8, 7), date(2026, 8, 21))


def test_select_term_expiries_extrapolates_one_sided(config: Config) -> None:
    day = 24 * 60.0
    # both beyond 30d -> two smallest chosen, ordered
    later = {
        date(2026, 8, 21): 43 * day,
        date(2026, 9, 18): 71 * day,
        date(2026, 12, 18): 162 * day,
    }
    assert select_term_expiries(later, config.implied) == (
        date(2026, 8, 21),
        date(2026, 9, 18),
    )
    with pytest.raises(ValueError, match="at least two expiries"):
        select_term_expiries({date(2026, 8, 21): 43 * day}, config.implied)


def test_settlement_conventions(config: Config) -> None:
    # third-Friday SPX is AM-settled 09:30 ET (13:30 UTC in July)
    standard = expiry_settlement("SPX", date(2026, 7, 17), config.implied)
    assert standard == datetime(2026, 7, 17, 13, 30, tzinfo=UTC)
    # weeklies and ETF options are PM-settled 16:00 ET
    weekly = expiry_settlement("SPXW", date(2026, 7, 17), config.implied)
    assert weekly == datetime(2026, 7, 17, 20, 0, tzinfo=UTC)
    fxe = expiry_settlement("FXE", date(2026, 8, 21), config.implied)
    assert fxe.hour == 20
    # minutes arithmetic
    asof = datetime(2026, 7, 16, 20, 0, tzinfo=UTC)
    assert minutes_to(asof, weekly) == pytest.approx(24 * 60.0)


def test_carry_forward_fallback_when_no_parity_pair(config: Config) -> None:
    """Thin ETF chains (near-term FXE) can lack any strike quoted both sides."""
    t = 30.0 / 365.0
    chain = synthetic_chain(t, np.arange(70.0, 140.0, 1.0))
    # zero every put bid at strikes >= 100 and every call bid below 100:
    # both wings survive but no strike has a two-sided call AND put
    chain = chain.with_columns(
        pl.when(
            ((pl.col("call_put") == "P") & (pl.col("strike") >= 100.0))
            | ((pl.col("call_put") == "C") & (pl.col("strike") < 100.0))
        )
        .then(0.0)
        .otherwise(pl.col("bid"))
        .alias("bid")
    )
    carry_forward = float(S * np.exp((R - Q) * t))
    with pytest.raises(ValueError, match="two-sided call and put"):
        single_expiry_variance(chain, t_years=t, r=R, cfg=config.implied)
    result = single_expiry_variance(
        chain, t_years=t, r=R, cfg=config.implied, fallback_forward=carry_forward
    )
    assert result.forward == carry_forward
    # OTM wings intact around K0, so the estimate still recovers σ² closely
    assert result.sigma_sq == pytest.approx(SIGMA**2, rel=2e-2)


def test_variance_errors_on_hopeless_chains(config: Config) -> None:
    t = 30.0 / 365.0
    empty = pl.DataFrame({"strike": [100.0], "call_put": ["C"], "bid": [1.0], "ask": [1.2]})
    with pytest.raises(ValueError, match="two-sided call and put"):
        single_expiry_variance(empty, t_years=t, r=R, cfg=config.implied)
    with pytest.raises(ValueError, match="t_years"):
        single_expiry_variance(empty, t_years=0.0, r=R, cfg=config.implied)
