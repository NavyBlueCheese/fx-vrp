"""VRP series construction: gap returns, leg alignment, and the zero identity."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from fxvrp.analysis.vrp import add_gap_returns, build_vrp_series, implied_leg
from fxvrp.config import Config


def _weekdays(start: date, n: int) -> list[date]:
    days = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _full_panel(rv: list[float], start: date = date(2018, 1, 1)) -> pl.DataFrame:
    """A panel with flat log-mids except a weekend jump before each Monday."""
    days = _weekdays(start, len(rv))
    first_mid, last_mid = [], []
    level = 0.0
    for day in days:
        if day.weekday() == 0:
            level += 0.01  # the weekend gap: Monday opens 1% above Friday's close
        first_mid.append(level)
        last_mid.append(level)  # flat inside the day
    return pl.DataFrame(
        {
            "day": days,
            "rv": rv,
            "rv_total": rv,
            "bpv": [v * 0.9 for v in rv],
            "bns_z": [0.0] * len(rv),
            "first_log_mid": first_mid,
            "last_log_mid": last_mid,
            "flag_thin": [False] * len(rv),
        }
    )


def test_gap_returns_capture_weekends_only() -> None:
    panel = _full_panel([1e-4] * 10)
    with_gaps = add_gap_returns(panel)
    first_day = with_gaps["day"][0]
    mondays = with_gaps.filter((pl.col("day").dt.weekday() == 1) & (pl.col("day") != first_day))
    others = with_gaps.filter(pl.col("day").dt.weekday() != 1)
    assert mondays.height >= 1
    assert np.allclose(mondays["gap_sq"].to_numpy(), 0.01**2)
    assert np.allclose(others["gap_sq"].to_numpy()[1:], 0.0)  # first row has no prior
    assert np.allclose(
        with_gaps["rv_total"].to_numpy(),
        with_gaps["rv"].to_numpy() + with_gaps["gap_sq"].to_numpy(),
    )


def test_implied_leg_squares_the_index() -> None:
    evz = pl.DataFrame({"date": [date(2018, 1, 1), date(2018, 1, 2)], "value": [10.0, None]})
    leg = implied_leg(evz)
    assert leg.height == 1  # null dropped, logged
    assert leg["iv_sq"][0] == pytest.approx(0.01)


def test_ex_post_vrp_is_zero_when_implied_equals_realized(config: Config) -> None:
    """The books-balance identity: EVZ² == future realised ⇒ VRP^{ex-post} = 0."""
    n = 260
    daily_var = 1e-5  # flat variance world, no gaps
    panel = _full_panel([daily_var] * n).with_columns(
        pl.lit(0.0).alias("first_log_mid"), pl.lit(0.0).alias("last_log_mid")
    )
    horizon = config.vrp.forward_window_calendar_days
    ann = config.vrp.annualize_days / horizon

    # per-day forward windows differ slightly (21 or 22 rows per 30 calendar
    # days), so hand each day exactly its own realised window as the "index"
    days = panel["day"].to_list()
    implied_values = []
    for i, day_t in enumerate(days):
        window_end = day_t + timedelta(days=horizon)
        total = sum(daily_var for d in days[i + 1 :] if d <= window_end)
        implied_values.append(100.0 * np.sqrt(total * ann))
    evz = pl.DataFrame({"date": days, "value": implied_values})

    series = build_vrp_series(panel, evz, config.vrp)
    ex_post = series.drop_nulls("vrp_ex_post")["vrp_ex_post"].to_numpy()
    assert ex_post.size > 100
    assert np.allclose(ex_post, 0.0, atol=1e-12)


def test_thin_days_are_excluded_and_logged(config: Config) -> None:
    panel = _full_panel([1e-5] * 30)
    panel = panel.with_columns(
        pl.when(pl.arange(0, panel.height) == 5)
        .then(True)
        .otherwise(pl.col("flag_thin"))
        .alias("flag_thin")
    )
    evz = pl.DataFrame({"date": panel["day"].to_list(), "value": [10.0] * panel.height})
    series = build_vrp_series(panel, evz, config.vrp)
    dropped_day = panel["day"][5]
    assert series.filter(pl.col("day") == dropped_day).height == 0
