"""The volatility signature plot: RV as a function of sampling interval.

Under microstructure noise, E[RV(Δ)] ≈ IV + 2nω² with n = T/Δ, so RV blows up
as Δ → 0 — the plot makes the bias visible and justifies the baseline interval
(Andersen, Bollerslev, Diebold & Labys 2000, "Great realizations", Risk 13).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from fxvrp.config import RealizedConfig
from fxvrp.realized.estimators import realized_variance
from fxvrp.realized.sampling import fx_day_window, grid_returns, previous_tick_grid


def signature_curve(
    ticks: pl.DataFrame,
    label: date,
    realized_cfg: RealizedConfig,
) -> pl.DataFrame:
    """(interval_s, rv, n_returns) for one FX day across the config intervals."""
    window = fx_day_window(label, realized_cfg.day_close_local, realized_cfg.day_close_tz)
    rows: list[dict[str, object]] = []
    for interval in realized_cfg.signature_intervals_s:
        prices = previous_tick_grid(ticks, window, interval)
        returns = grid_returns(prices)
        rows.append(
            {
                "day": label,
                "interval_s": interval,
                "rv": float(realized_variance(returns)) if returns.size else None,
                "n_returns": int(returns.size),
            }
        )
    return pl.DataFrame(rows)


def average_signature(curves: list[pl.DataFrame]) -> pl.DataFrame:
    """Average per-interval RV across days (days with data at that interval)."""
    stacked = pl.concat(curves)
    return (
        stacked.drop_nulls("rv")
        .group_by("interval_s")
        .agg(pl.col("rv").mean().alias("mean_rv"), pl.len().alias("n_days"))
        .sort("interval_s")
    )
