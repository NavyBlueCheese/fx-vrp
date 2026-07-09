"""Short-rate utilities: fed funds → continuously-compounded discount rate.

Conventions.md rule 10: DFF is an annualised simple ACT/360 rate; one day's
growth is (1 + R/360), so the continuously-compounded ACT/365 equivalent is
r = 365 · ln(1 + R/360). For the ≤60-day tenors priced here, the flat-overnight
proxy for the term rate is accepted and disclosed.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl

_DAYS_MONEY_MARKET = 360.0
_DAYS_CC = 365.0
_PCT = 100.0


def cc_rate_from_fed_funds(rate_pct: float) -> float:
    """Continuously-compounded ACT/365 rate from an ACT/360 simple percent quote."""
    return _DAYS_CC * math.log(1.0 + rate_pct / _PCT / _DAYS_MONEY_MARKET)


def rate_asof(frame: pl.DataFrame, day: date) -> float:
    """Last observed rate at or before ``day`` from a (date, value) frame."""
    prior = frame.drop_nulls("value").filter(pl.col("date") <= day)
    if prior.height == 0:
        raise ValueError(f"no rate observation at or before {day}")
    value = prior.sort("date")["value"][-1]
    return float(value)
