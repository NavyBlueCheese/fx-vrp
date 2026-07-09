"""Tick data-quality metrics: the numbers behind the data-quality report.

Phase 0 flagged the known Dukascopy pathologies — gaps, stale quotes, crossed
quotes around rollovers and holidays. This module measures them per day so the
appendix report can show them rather than assert their absence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from fxvrp.config import QualityConfig


@dataclass(frozen=True)
class DayQuality:
    day: date
    n_ticks: int
    median_spread_pips: float
    p95_spread_pips: float
    n_crossed: int  # ask <= bid
    n_wide: int  # spread beyond plausibility bound
    max_gap_s: float  # longest intra-day quote silence
    n_gaps_reportable: float  # gaps longer than the report threshold
    max_stale_run_s: float  # longest run with both bid and ask unchanged
    flag_low_ticks: bool


def day_quality(frame: pl.DataFrame, day: date, cfg: QualityConfig, pip: float) -> DayQuality:
    """Compute per-day quality metrics from a (ts, bid, ask) tick frame. Pure."""
    if frame.height == 0:
        return DayQuality(
            day=day,
            n_ticks=0,
            median_spread_pips=float("nan"),
            p95_spread_pips=float("nan"),
            n_crossed=0,
            n_wide=0,
            max_gap_s=float("nan"),
            n_gaps_reportable=0,
            max_stale_run_s=float("nan"),
            flag_low_ticks=True,
        )

    enriched = (
        frame.sort("ts")
        .with_columns(((pl.col("ask") - pl.col("bid")) / pip).alias("spread_pips"))
        .with_columns(
            (pl.col("ts").diff().dt.total_milliseconds() / 1_000.0).alias("gap_s"),
            (
                (pl.col("bid") != pl.col("bid").shift(1))
                | (pl.col("ask") != pl.col("ask").shift(1))
            )
            .fill_null(True)
            .alias("quote_changed"),
        )
        .with_columns(pl.col("quote_changed").cum_sum().alias("stale_group"))
    )

    stale = (
        enriched.group_by("stale_group")
        .agg((pl.col("ts").max() - pl.col("ts").min()).dt.total_milliseconds().alias("run_ms"))
        .select(pl.col("run_ms").max())
        .item()
    )
    gaps = enriched["gap_s"].drop_nulls()

    return DayQuality(
        day=day,
        n_ticks=frame.height,
        median_spread_pips=_as_float(enriched["spread_pips"].median()),
        p95_spread_pips=_as_float(enriched["spread_pips"].quantile(0.95, interpolation="linear")),
        n_crossed=int(enriched.filter(pl.col("spread_pips") <= 0.0).height),
        n_wide=int(
            enriched.filter(pl.col("spread_pips") > cfg.max_plausible_spread_pips).height
        ),
        max_gap_s=_as_float(gaps.max()) if gaps.len() > 0 else float("nan"),
        n_gaps_reportable=int(gaps.filter(gaps > cfg.max_gap_report_s).len())
        if gaps.len() > 0
        else 0,
        max_stale_run_s=_as_float(stale) / 1_000.0,
        flag_low_ticks=frame.height < cfg.min_ticks_per_day,
    )


def _as_float(value: object) -> float:
    """Narrow polars aggregate results (broadly-typed unions) to float."""
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def quality_table(results: list[DayQuality]) -> pl.DataFrame:
    """Stack per-day metrics into a report-ready frame."""
    return pl.DataFrame(
        [
            {
                "day": r.day,
                "n_ticks": r.n_ticks,
                "median_spread_pips": r.median_spread_pips,
                "p95_spread_pips": r.p95_spread_pips,
                "n_crossed": r.n_crossed,
                "n_wide": r.n_wide,
                "max_gap_s": r.max_gap_s,
                "n_gaps_reportable": r.n_gaps_reportable,
                "max_stale_run_s": r.max_stale_run_s,
                "flag_low_ticks": r.flag_low_ticks,
            }
            for r in results
        ]
    )
