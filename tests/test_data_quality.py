from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from fxvrp.config import Config
from fxvrp.data.quality import day_quality, quality_table

DAY = date(2020, 3, 9)
T0 = datetime(2020, 3, 9, 0, 0, tzinfo=UTC)


def _ticks(offsets_s: list[float], bids: list[float], asks: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [T0 + timedelta(seconds=s) for s in offsets_s],
            "bid": bids,
            "ask": asks,
        },
        schema_overrides={"ts": pl.Datetime(time_unit="ms", time_zone="UTC")},
    )


def test_day_quality_measures_spread_gap_stale_and_crossed(config: Config) -> None:
    frame = _ticks(
        offsets_s=[0.0, 1.0, 2.0, 200.0, 700.0],
        bids=[1.1000, 1.1001, 1.1001, 1.1002, 1.1003],
        asks=[1.1001, 1.1002, 1.1001, 1.1003, 1.1004],  # third quote crossed (locked)
    )
    quality = day_quality(frame, DAY, config.quality, config.dukascopy.pip)

    assert quality.n_ticks == 5
    assert quality.n_crossed == 1
    assert quality.median_spread_pips == pytest.approx(1.0)
    assert quality.max_gap_s == pytest.approx(500.0)
    assert quality.n_gaps_reportable == 2  # gaps are 1, 1, 198, 500 s -> two exceed 120s
    assert quality.flag_low_ticks  # 5 < configured minimum


def test_day_quality_stale_run_is_longest_unchanged_quote(config: Config) -> None:
    frame = _ticks(
        offsets_s=[0.0, 100.0, 400.0, 401.0],
        bids=[1.1000, 1.1000, 1.1000, 1.1001],
        asks=[1.1001, 1.1001, 1.1001, 1.1002],
    )
    quality = day_quality(frame, DAY, config.quality, config.dukascopy.pip)
    assert quality.max_stale_run_s == pytest.approx(400.0)


def test_day_quality_empty_day_is_flagged(config: Config) -> None:
    frame = _ticks([], [], [])
    quality = day_quality(frame, DAY, config.quality, config.dukascopy.pip)
    assert quality.n_ticks == 0
    assert quality.flag_low_ticks


def test_quality_table_stacks_days(config: Config) -> None:
    frame = _ticks([0.0, 1.0], [1.1, 1.1], [1.1001, 1.1001])
    table = quality_table([day_quality(frame, DAY, config.quality, config.dukascopy.pip)])
    assert table.height == 1
    assert "median_spread_pips" in table.columns
