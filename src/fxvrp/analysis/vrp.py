"""The variance risk premium series — two objects, never confused.

    VRP^{ex-post}_t = IV²_t − RV_{t,t+30}      (uses the future: DESCRIPTIVE ONLY)
    VRP^{ex-ante}_t = IV²_t − E_t[RV_{t,t+30}] (HAR forecast: the tradeable object)

Both legs are annualised 30-calendar-day variances (conventions.md rule 3):
the implied leg is (EVZ/100)², the realised leg is the summed daily panel over
(t, t+30d] × 365/30, with inter-day gap returns (weekends, holidays) added per
rule 5 — the implied side charges for calendar time, so the realised side must
pay for it.

Alignment note (rule 4, amended): EVZ prints at 16:15 ET; the realised window
runs on 17:00-ET FX days. The 45-minute mismatch is ~0.1% of the window and is
disclosed, not corrected.
"""

from __future__ import annotations

import polars as pl

from fxvrp.config import VrpConfig
from fxvrp.log import get_logger, log_filter
from fxvrp.realized.har import (
    continuous_jump_split,
    feature_columns,
    forward_realized_window,
    har_features,
    walk_forward_forecast,
)

logger = get_logger("analysis.vrp")

_PCT = 100.0


def add_gap_returns(panel: pl.DataFrame) -> pl.DataFrame:
    """rv_total = intraday RV + the squared close-to-open gap versus the prior day.

    The gap between day d-1's last in-window quote and day d's first quote is
    ~0 on continuous weekdays and carries the weekend/holiday variance
    otherwise. Attributed to day d (the first window that could see it).
    """
    frame = panel.sort("day")
    return (
        frame.with_columns(
            (pl.col("first_log_mid") - pl.col("last_log_mid").shift(1)).alias("_gap")
        )
        .with_columns(
            (pl.col("_gap") ** 2).fill_null(0.0).alias("gap_sq"),
            (pl.col("rv") + (pl.col("_gap") ** 2).fill_null(0.0)).alias("rv_total"),
        )
        .drop("_gap")
    )


def implied_leg(evz: pl.DataFrame) -> pl.DataFrame:
    """(day, iv_sq): annualised 30-day implied variance from the EVZ close."""
    out = (
        evz.drop_nulls("value")
        .select(
            pl.col("date").alias("day"),
            ((pl.col("value") / _PCT) ** 2).alias("iv_sq"),
        )
        .sort("day")
    )
    log_filter(logger, "vrp.implied_leg", evz.height, out.height, "EVZ value is null")
    return out


def build_vrp_series(
    panel: pl.DataFrame,
    evz: pl.DataFrame,
    cfg: VrpConfig,
    *,
    jump_split: bool = False,
) -> pl.DataFrame:
    """Assemble the daily VRP frame: both legs, both VRP definitions.

    Columns: day, iv_sq, rv_fwd_ann (ex-post leg), rv_fwd_hat_ann (HAR
    forecast), vrp_ex_post, vrp_ex_ante, window_coverage, plus the HAR features
    for later conditional analysis. Days where the panel is thin are excluded
    (logged) before anything else.
    """
    n_in = panel.height
    usable = panel.filter(~pl.col("flag_thin"))
    log_filter(logger, "vrp.thin_days", n_in, usable.height, "flag_thin")

    with_gaps = add_gap_returns(usable)
    if jump_split:
        with_gaps = continuous_jump_split(with_gaps, cfg.jump_alpha)
    features = har_features(with_gaps, cfg.har_lags, jump_split=jump_split)
    with_target = forward_realized_window(features, cfg)

    forecast = walk_forward_forecast(
        with_target,
        cfg,
        feature_cols=feature_columns(cfg.har_lags, jump_split),
    )
    joined = (
        with_target.join(forecast, on="day", how="left")
        .join(implied_leg(evz), on="day", how="inner")
        .with_columns(
            (pl.col("iv_sq") - pl.col("rv_fwd_ann")).alias("vrp_ex_post"),
            (pl.col("iv_sq") - pl.col("rv_fwd_hat_ann")).alias("vrp_ex_ante"),
        )
        .sort("day")
    )
    logger.info(
        "VRP series: %d days joined, %d with ex-post, %d with ex-ante",
        joined.height,
        joined.drop_nulls("vrp_ex_post").height,
        joined.drop_nulls("vrp_ex_ante").height,
    )
    return joined
