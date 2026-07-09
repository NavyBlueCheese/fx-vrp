from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from fxvrp.config import Config
from fxvrp.data.chains import clean_chain

ASOF = date(2026, 7, 8)
TS = datetime(2026, 7, 8, 20, 0, tzinfo=UTC)


def _snapshot() -> pl.DataFrame:
    rows = [
        # good two-sided quote, 30d out
        ("FXE260807C00105000", date(2026, 8, 7), "C", 105.0, 1.00, 1.10),
        # zero bid -> dropped
        ("FXE260807P00095000", date(2026, 8, 7), "P", 95.0, 0.00, 0.30),
        # crossed -> dropped
        ("FXE260807C00106000", date(2026, 8, 7), "C", 106.0, 1.20, 1.10),
        # expires today -> dropped by expiry window
        ("FXE260708C00105000", date(2026, 7, 8), "C", 105.0, 0.50, 0.60),
        # far LEAP beyond window -> dropped
        ("FXE280121C00105000", date(2028, 1, 21), "C", 105.0, 2.00, 2.40),
        # duplicate of the good contract -> deduplicated
        ("FXE260807C00105000", date(2026, 8, 7), "C", 105.0, 1.01, 1.11),
    ]
    return pl.DataFrame(
        {
            "underlying": ["FXE"] * len(rows),
            "contract": [r[0] for r in rows],
            "expiry": [r[1] for r in rows],
            "call_put": [r[2] for r in rows],
            "strike": [r[3] for r in rows],
            "bid": [r[4] for r in rows],
            "ask": [r[5] for r in rows],
            "quote_time": [TS] * len(rows),
        }
    )


def test_clean_chain_applies_every_rule_and_accounts_for_every_row(config: Config) -> None:
    result = clean_chain(_snapshot(), config.chain_cleaning, ASOF)

    assert result.n_input == 6
    assert result.frame.height == 1
    assert result.dropped == {
        "not_two_sided": 2,
        "expiry_window": 2,
        "duplicate_contract": 1,
    }
    # full reconciliation: input = output + all drops
    assert result.n_input == result.frame.height + sum(result.dropped.values())

    survivor = result.frame.row(0, named=True)
    assert survivor["contract"] == "FXE260807C00105000"
    assert survivor["bid"] == pytest.approx(1.00)  # first occurrence kept
    assert survivor["mid"] == pytest.approx(1.05)
    assert survivor["spread"] == pytest.approx(0.10)
    assert survivor["days_to_expiry"] == 30


def test_clean_chain_empty_input(config: Config) -> None:
    empty = _snapshot().head(0)
    result = clean_chain(empty, config.chain_cleaning, ASOF)
    assert result.frame.height == 0
    assert sum(result.dropped.values()) == 0
