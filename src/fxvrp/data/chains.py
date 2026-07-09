"""Chain cleaning: quote validation with fully-logged, reconcilable drops.

Rules (configs/default.yaml `chain_cleaning`, conventions.md rule 16):
  1. two-sided quotes only — bid > min_bid and ask > bid (zero-bid wings and
     crossed/locked markets carry no usable price);
  2. expiry window — days to expiry within [min_days_to_expiry, max_days_to_expiry]
     as of the snapshot date;
  3. duplicate contracts — the same contract twice in one snapshot keeps the first
     occurrence.

Every rule logs rows in / rows out / reason. The dropped counts are returned so
callers can persist them alongside the cleaned data; reconstruction of exactly what
was dropped and why is a hard project requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from fxvrp.config import ChainCleaningConfig
from fxvrp.log import get_logger, log_filter

logger = get_logger("data.chains")


@dataclass(frozen=True)
class CleanChainResult:
    frame: pl.DataFrame
    n_input: int
    dropped: dict[str, int]  # reason -> rows removed


def clean_chain(
    frame: pl.DataFrame,
    cfg: ChainCleaningConfig,
    asof: date,
) -> CleanChainResult:
    """Apply the quote-validation rules to one normalized chain snapshot."""
    n_input = frame.height
    dropped: dict[str, int] = {}

    step = frame.filter((pl.col("bid") > cfg.min_bid) & (pl.col("ask") > pl.col("bid")))
    dropped["not_two_sided"] = n_input - step.height
    log_filter(
        logger,
        "chains.two_sided",
        n_input,
        step.height,
        f"bid <= {cfg.min_bid} or ask <= bid",
    )

    n_before = step.height
    step = step.with_columns(
        (pl.col("expiry") - pl.lit(asof)).dt.total_days().alias("days_to_expiry")
    ).filter(pl.col("days_to_expiry").is_between(cfg.min_days_to_expiry, cfg.max_days_to_expiry))
    dropped["expiry_window"] = n_before - step.height
    log_filter(
        logger,
        "chains.expiry_window",
        n_before,
        step.height,
        f"days_to_expiry outside [{cfg.min_days_to_expiry}, {cfg.max_days_to_expiry}]",
    )

    n_before = step.height
    step = step.unique(subset=["contract"], keep="first", maintain_order=True)
    dropped["duplicate_contract"] = n_before - step.height
    log_filter(logger, "chains.duplicates", n_before, step.height, "duplicate contract id")

    cleaned = step.with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
    )
    return CleanChainResult(frame=cleaned, n_input=n_input, dropped=dropped)
