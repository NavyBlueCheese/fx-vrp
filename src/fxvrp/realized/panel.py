"""The daily realised-variance panel: raw tick parquets → one row per FX day.

Each FX day (17:00-ET close convention) draws its ticks from the two calendar-UTC
day files that straddle its window. Days with too few grid returns keep their row
with null estimators and a flag — a thin day is information, not a gap to hide.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from fxvrp.config import DukascopyConfig, RealizedConfig
from fxvrp.data.dukascopy import day_parquet_path
from fxvrp.log import get_logger
from fxvrp.realized.estimators import realized_semivariance, realized_variance
from fxvrp.realized.jumps import (
    bipower_variation,
    bns_test_statistic,
    jump_variation,
    tripower_quarticity,
)
from fxvrp.realized.sampling import (
    fx_day_window,
    grid_returns,
    previous_tick_grid,
    window_ticks,
)

logger = get_logger("realized.panel")

_SATURDAY = 5  # date.weekday(): Monday=0 .. Saturday=5

PANEL_SCHEMA: dict[str, pl.DataType] = {
    "day": pl.Date(),
    "n_ticks": pl.Int64(),
    "n_returns": pl.Int64(),
    "rv": pl.Float64(),
    "bpv": pl.Float64(),
    "jv": pl.Float64(),
    "tq": pl.Float64(),
    "bns_z": pl.Float64(),
    "rs_plus": pl.Float64(),
    "rs_minus": pl.Float64(),
    "signed_jump": pl.Float64(),
    # first/last in-window log mid: the 30-day aggregator uses consecutive-day
    # (last, first) pairs to add weekend/holiday gap returns (conventions rule 5),
    # which no single intraday window can see
    "first_log_mid": pl.Float64(),
    "last_log_mid": pl.Float64(),
    "flag_thin": pl.Boolean(),
}


@dataclass(frozen=True)
class PanelBuildResult:
    frame: pl.DataFrame
    n_days: int
    n_thin: int


def _load_window_ticks(raw_dir: Path, instrument: str, label: date) -> pl.DataFrame | None:
    """Ticks for the FX day ``label``: the two straddling calendar-day files."""
    frames = []
    for calendar_day in (label - timedelta(days=1), label):
        path = day_parquet_path(raw_dir, instrument, calendar_day)
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return None
    return pl.concat(frames).sort("ts")


def build_day_row(
    ticks: pl.DataFrame,
    label: date,
    realized_cfg: RealizedConfig,
) -> dict[str, object]:
    """Estimator row for one FX day. Pure given the day's ticks.

    Thinness is judged on *quote updates inside the window*, not on grid
    returns: previous-tick sampling holds the last quote, so even a single tick
    fills the whole grid — with zeros that would silently deflate every
    estimator. A day with fewer updates than required grid returns is flagged
    and left null.
    """
    window = fx_day_window(label, realized_cfg.day_close_local, realized_cfg.day_close_tz)
    in_window = window_ticks(ticks, window)
    prices = previous_tick_grid(in_window, window, realized_cfg.grid_interval_s)
    returns = grid_returns(prices)
    n_returns = int(returns.size)

    row: dict[str, object] = {
        "day": label,
        "n_ticks": in_window.height,
        "n_returns": n_returns,
        "rv": None,
        "bpv": None,
        "jv": None,
        "tq": None,
        "bns_z": None,
        "rs_plus": None,
        "rs_minus": None,
        "signed_jump": None,
        "first_log_mid": float(prices[0]) if prices.size else None,
        "last_log_mid": float(prices[-1]) if prices.size else None,
        "flag_thin": True,
    }
    if (
        in_window.height < realized_cfg.min_returns_per_day
        or n_returns < realized_cfg.min_returns_per_day
    ):
        return row

    rs_plus, rs_minus = realized_semivariance(returns)
    row.update(
        rv=float(realized_variance(returns)),
        bpv=float(bipower_variation(returns)),
        jv=float(jump_variation(returns)),
        tq=float(tripower_quarticity(returns)),
        bns_z=bns_test_statistic(returns),
        rs_plus=float(rs_plus),
        rs_minus=float(rs_minus),
        signed_jump=float(rs_plus) - float(rs_minus),
        flag_thin=False,
    )
    return row


def build_panel(
    raw_dir: Path,
    duka_cfg: DukascopyConfig,
    realized_cfg: RealizedConfig,
    start: date,
    end: date,
) -> PanelBuildResult:
    """Build the daily panel over [start, end] from whatever ticks are on disk.

    FX days run Monday..Friday by close label (a Sunday-reopen belongs to
    Monday's window). Days with no tick files at all are skipped and counted —
    absence of a file is an ingestion gap, not a market holiday claim.
    """
    rows: list[dict[str, object]] = []
    n_missing = 0
    label = start
    while label <= end:
        if label.weekday() < _SATURDAY:  # Monday..Friday close labels
            ticks = _load_window_ticks(raw_dir, duka_cfg.instrument, label)
            if ticks is None:
                n_missing += 1
            else:
                rows.append(build_day_row(ticks, label, realized_cfg))
        label += timedelta(days=1)

    frame = pl.DataFrame(rows, schema_overrides=PANEL_SCHEMA).sort("day")
    n_thin = int(frame.filter(pl.col("flag_thin")).height) if frame.height else 0
    logger.info(
        "panel %s..%s: %d days built, %d thin, %d without tick files",
        start,
        end,
        frame.height,
        n_thin,
        n_missing,
    )
    return PanelBuildResult(frame=frame, n_days=frame.height, n_thin=n_thin)
