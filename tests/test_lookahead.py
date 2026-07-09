"""The lookahead guard, including the deliberate-contamination case (brief §4.7)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from fxvrp.lookahead import LookaheadError, assert_information_set


def _frame(days: list[date]) -> pl.DataFrame:
    return pl.DataFrame({"day": days, "x": list(range(len(days)))})


def test_clean_information_set_passes_through() -> None:
    frame = _frame([date(2020, 1, 2), date(2020, 1, 3)])
    out = assert_information_set(frame, asof=date(2020, 1, 3))
    assert out is frame  # identity: the guard is a checkpoint, not a copy


def test_future_data_raises() -> None:
    frame = _frame([date(2020, 1, 2), date(2020, 1, 6), date(2020, 1, 7)])
    with pytest.raises(LookaheadError, match=r"2 row.*from the future"):
        assert_information_set(frame, asof=date(2020, 1, 3))


def test_boundary_is_inclusive() -> None:
    frame = _frame([date(2020, 1, 3)])
    assert_information_set(frame, asof=date(2020, 1, 3))  # same-day info is admissible
    with pytest.raises(LookaheadError):
        assert_information_set(frame, asof=date(2020, 1, 2))


def test_null_timestamps_are_refused() -> None:
    frame = pl.DataFrame({"day": [date(2020, 1, 2), None], "x": [1, 2]})
    with pytest.raises(LookaheadError, match="null timestamps"):
        assert_information_set(frame, asof=date(2020, 1, 3))


def test_missing_timestamp_column_is_refused() -> None:
    with pytest.raises(LookaheadError, match="no timestamp column"):
        assert_information_set(pl.DataFrame({"x": [1]}), asof=date(2020, 1, 3))
