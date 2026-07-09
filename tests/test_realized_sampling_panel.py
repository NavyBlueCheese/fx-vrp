"""Sampling conventions and the daily panel builder, on crafted tick data."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fxvrp.config import Config
from fxvrp.data.dukascopy import day_parquet_path
from fxvrp.realized.panel import build_day_row, build_panel
from fxvrp.realized.sampling import fx_day_window, grid_returns, previous_tick_grid
from fxvrp.realized.signature import average_signature, signature_curve

LABEL = date(2020, 3, 10)  # a Tuesday


def _tick_frame(times: list[datetime], mids: list[float], half_spread: float = 0.00005):
    return pl.DataFrame(
        {
            "ts": times,
            "bid": [m - half_spread for m in mids],
            "ask": [m + half_spread for m in mids],
        },
        schema_overrides={"ts": pl.Datetime(time_unit="ms", time_zone="UTC")},
    )


def test_fx_day_window_is_ny_17h_dst_aware(config: Config) -> None:
    cfg = config.realized
    # March 2020: US already on EDT (UTC-4) -> 17:00 ET == 21:00 UTC
    summer = fx_day_window(date(2020, 3, 10), cfg.day_close_local, cfg.day_close_tz)
    assert summer.end.astimezone(UTC).hour == 21
    # January: EST (UTC-5) -> 22:00 UTC
    winter = fx_day_window(date(2020, 1, 10), cfg.day_close_local, cfg.day_close_tz)
    assert winter.end.astimezone(UTC).hour == 22
    assert (summer.end - summer.start) == timedelta(days=1)


def test_previous_tick_carries_last_quote(config: Config) -> None:
    cfg = config.realized
    window = fx_day_window(LABEL, cfg.day_close_local, cfg.day_close_tz)
    t0 = window.start
    ticks = _tick_frame(
        [t0 + timedelta(minutes=1), t0 + timedelta(minutes=7), t0 + timedelta(minutes=21)],
        [1.10, 1.11, 1.12],
    )
    prices = previous_tick_grid(ticks, window, interval_s=300)
    # grid at +5m sees the 1.10 quote; +10m..+20m carry 1.11; +25m onward 1.12
    assert prices[0] == pytest.approx(np.log(1.10))
    assert prices[1] == pytest.approx(np.log(1.11))
    assert prices[3] == pytest.approx(np.log(1.11))
    assert prices[4] == pytest.approx(np.log(1.12))
    # the carried quote is *held*, so later grid points repeat it (zero return)
    assert prices[-1] == pytest.approx(np.log(1.12))
    returns = grid_returns(prices)
    assert (returns == 0.0).sum() > 200  # long quiet stretch: held quotes


def test_previous_tick_ignores_out_of_window_ticks(config: Config) -> None:
    cfg = config.realized
    window = fx_day_window(LABEL, cfg.day_close_local, cfg.day_close_tz)
    ticks = _tick_frame([window.start - timedelta(hours=1)], [1.05])  # before the window
    assert previous_tick_grid(ticks, window, 300).size == 0
    assert previous_tick_grid(ticks.head(0), window, 300).size == 0


def test_build_day_row_flags_thin_days(config: Config) -> None:
    window = fx_day_window(LABEL, config.realized.day_close_local, config.realized.day_close_tz)
    ticks = _tick_frame([window.start + timedelta(minutes=5)], [1.10])
    row = build_day_row(ticks, LABEL, config.realized)
    assert row["flag_thin"] is True
    assert row["rv"] is None


def test_build_day_row_estimates_a_dense_day(config: Config) -> None:
    rng = np.random.default_rng(7)
    window = fx_day_window(LABEL, config.realized.day_close_local, config.realized.day_close_tz)
    n = 5_000
    times = [window.start + timedelta(seconds=17.28 * k) for k in range(n)]
    mids = list(1.10 * np.exp(np.cumsum(rng.standard_normal(n) * 2e-5)))
    row = build_day_row(_tick_frame(times, mids), LABEL, config.realized)
    assert row["flag_thin"] is False
    assert row["rv"] is not None and float(str(row["rv"])) > 0.0
    rv, rsp, rsm = float(str(row["rv"])), float(str(row["rs_plus"])), float(str(row["rs_minus"]))
    assert rsp + rsm == pytest.approx(rv, rel=1e-9)
    assert float(str(row["jv"])) >= 0.0


def test_build_panel_reads_straddling_files_and_reports_gaps(
    tmp_path: Path, config: Config
) -> None:
    rng = np.random.default_rng(3)
    window = fx_day_window(LABEL, config.realized.day_close_local, config.realized.day_close_tz)
    n = 4_000
    times = [window.start + timedelta(seconds=21.6 * k) for k in range(n)]
    mids = list(1.10 * np.exp(np.cumsum(rng.standard_normal(n) * 2e-5)))
    ticks = _tick_frame(times, mids).with_columns(pl.col("ts").dt.convert_time_zone("UTC"))

    # split ticks into the two straddling calendar-UTC files, as ingestion would
    for calendar_day in (LABEL - timedelta(days=1), LABEL):
        day_start = datetime(calendar_day.year, calendar_day.month, calendar_day.day, tzinfo=UTC)
        chunk = ticks.filter(
            (pl.col("ts") >= day_start) & (pl.col("ts") < day_start + timedelta(days=1))
        )
        path = day_parquet_path(tmp_path, config.dukascopy.instrument, calendar_day)
        path.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_parquet(path)

    result = build_panel(
        tmp_path, config.dukascopy, config.realized, LABEL, LABEL + timedelta(days=1)
    )
    # LABEL is dense; LABEL+1 finds a straddling file but no in-window ticks -> thin
    assert result.n_days == 2
    built = result.frame.row(0, named=True)
    assert built["day"] == LABEL
    assert built["flag_thin"] is False
    next_day = result.frame.row(1, named=True)
    assert next_day["flag_thin"] is True
    assert next_day["n_ticks"] == 0


def test_signature_curve_shows_noise_blowup(config: Config) -> None:
    """Bounced quotes must push fine-interval RV far above coarse-interval RV."""
    rng = np.random.default_rng(5)
    window = fx_day_window(LABEL, config.realized.day_close_local, config.realized.day_close_tz)
    n = 17_280  # one tick per 5 seconds
    times = [window.start + timedelta(seconds=5.0 * k) for k in range(n)]
    efficient = 1.10 * np.exp(np.cumsum(rng.standard_normal(n) * 1.2e-5))
    bounce = np.exp(rng.choice([-1.0, 1.0], size=n) * 8e-5)
    curve = signature_curve(_tick_frame(times, list(efficient * bounce)), LABEL, config.realized)
    table = average_signature([curve]).drop_nulls("mean_rv")
    finest = table.filter(pl.col("interval_s") == pl.col("interval_s").min())["mean_rv"][0]
    coarsest = table.filter(pl.col("interval_s") == pl.col("interval_s").max())["mean_rv"][0]
    assert finest > 3.0 * coarsest
