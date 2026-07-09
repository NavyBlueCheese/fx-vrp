"""Calendar-time sampling of tick data and the FX trading-day calendar.

Sampling scheme: **previous-tick** interpolation on log mid-quotes — the grid
price at time t_j is the last observed quote at or before t_j (Hansen & Lunde
2006, J. Business & Economic Statistics 24(2), recommend previous-tick over
linear interpolation, which induces spurious smoothness). This is a sampling
convention, not data fabrication: no quote is invented, the prevailing one is
carried (conventions.md rule 19 note).

Day convention (conventions.md rule 6): the FX day labelled D covers
(D-1 17:00, D 17:00] New York time, DST-aware. Mid = (bid+ask)/2; the spread
never enters realised variance — it belongs to the cost model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from fxvrp._types import FloatArray


@dataclass(frozen=True)
class DayWindow:
    label: date  # the day whose 17:00-ET close ends the window
    start: datetime  # exclusive
    end: datetime  # inclusive


def fx_day_window(label: date, close_local: str, tz_name: str) -> DayWindow:
    """The (start, end] window of the FX day labelled ``label``."""
    tz = ZoneInfo(tz_name)
    close_t = time.fromisoformat(close_local)
    end = datetime.combine(label, close_t, tzinfo=tz)
    start = datetime.combine(label - timedelta(days=1), close_t, tzinfo=tz)
    return DayWindow(label=label, start=start, end=end)


def log_mid(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach log mid-quote to a (ts, bid, ask) tick frame."""
    return frame.with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).log().alias("log_mid"))


def window_ticks(frame: pl.DataFrame, window: DayWindow) -> pl.DataFrame:
    """Ticks strictly inside the FX day window, compared in UTC."""
    if frame.height == 0:
        return frame
    start_utc = window.start.astimezone(UTC)
    end_utc = window.end.astimezone(UTC)
    return frame.filter((pl.col("ts") > start_utc) & (pl.col("ts") <= end_utc))


def previous_tick_grid(
    frame: pl.DataFrame,
    window: DayWindow,
    interval_s: int,
) -> FloatArray:
    """Log mid-quotes sampled at ``interval_s`` on a previous-tick basis.

    Grid points are window.start + k·interval for k = 1..K with K·interval
    spanning the window; a grid point earlier than the first tick is dropped
    (no price is invented before the first observation). Returns the sampled
    log-price vector; the caller differences it.
    """
    if frame.height == 0:
        return np.empty(0, dtype=np.float64)

    # compare in UTC: tick timestamps are stored UTC and polars refuses
    # cross-timezone datetime comparisons
    start_utc = window.start.astimezone(UTC)
    end_utc = window.end.astimezone(UTC)
    ticks = log_mid(frame).filter((pl.col("ts") > start_utc) & (pl.col("ts") <= end_utc)).sort("ts")
    if ticks.height == 0:
        return np.empty(0, dtype=np.float64)

    n_intervals = int((end_utc - start_utc).total_seconds()) // interval_s
    grid_times = pl.DataFrame(
        {"ts": [start_utc + timedelta(seconds=interval_s * (k + 1)) for k in range(n_intervals)]}
    ).with_columns(pl.col("ts").cast(ticks.schema["ts"]))

    sampled = grid_times.join_asof(
        ticks.select("ts", "log_mid"), on="ts", strategy="backward"
    ).drop_nulls("log_mid")
    return sampled["log_mid"].to_numpy().astype(np.float64)


_MIN_GRID_POINTS = 2  # differencing needs two prices


def grid_returns(log_prices: FloatArray) -> FloatArray:
    """Log returns of a sampled grid."""
    if log_prices.size < _MIN_GRID_POINTS:
        return np.empty(0, dtype=np.float64)
    return np.diff(log_prices)
