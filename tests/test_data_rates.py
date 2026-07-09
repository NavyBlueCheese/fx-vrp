from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from fxvrp.data.rates import cc_rate_from_fed_funds, rate_asof


def test_cc_conversion_matches_daily_compounding() -> None:
    r_cc = cc_rate_from_fed_funds(3.63)
    # growing at DFF for one ACT/360 day must equal e^{r_cc/365}
    assert math.exp(r_cc / 365.0) == pytest.approx(1.0 + 0.0363 / 360.0, rel=1e-12)
    assert r_cc == pytest.approx(0.0368, abs=2e-4)
    assert cc_rate_from_fed_funds(0.0) == 0.0


def test_rate_asof_uses_last_prior_observation() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2026, 7, 3), date(2026, 7, 6), date(2026, 7, 7)],
            "value": [3.60, None, 3.63],
        }
    )
    assert rate_asof(frame, date(2026, 7, 7)) == 3.63
    assert rate_asof(frame, date(2026, 7, 6)) == 3.60  # null skipped
    assert rate_asof(frame, date(2026, 7, 5)) == 3.60
    with pytest.raises(ValueError):
        rate_asof(frame, date(2026, 7, 1))
