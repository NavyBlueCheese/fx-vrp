"""The lookahead guard: contaminated information sets raise, loudly.

Any tradeable signal may use only information timestamped at or before its
decision time (conventions.md rule 18). Every walk-forward fit and every
backtest feature pull passes its training data through ``assert_information_set``
— so feeding the future in is not a silent bias but a crash. The guard itself
is under test, including the deliberate-contamination case.
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl


class LookaheadError(RuntimeError):
    """An information set contains data from after the decision time."""


def assert_information_set(
    frame: pl.DataFrame,
    *,
    asof: date | datetime,
    ts_col: str = "day",
) -> pl.DataFrame:
    """Return ``frame`` unchanged iff every row is timestamped <= ``asof``.

    Null timestamps are refused too: a row that cannot prove when it was known
    is not admissible evidence.
    """
    if ts_col not in frame.columns:
        raise LookaheadError(f"information set has no timestamp column {ts_col!r}")
    if frame[ts_col].null_count() > 0:
        raise LookaheadError(f"information set has {frame[ts_col].null_count()} null timestamps")
    future = frame.filter(pl.col(ts_col) > asof)
    if future.height > 0:
        first_bad = future[ts_col].min()
        raise LookaheadError(
            f"information set for decision time {asof} contains {future.height} row(s) "
            f"from the future (earliest: {first_bad!r})"
        )
    return frame
