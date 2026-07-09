"""FRED daily series via the keyless ``fredgraph.csv`` endpoint.

Format: two columns, ``observation_date,{SERIES_ID}``; missing observations are
".". EVZCLS is DISCONTINUED (last observation 2025-03-11) and must never be
extended or filled — the discontinuation is a finding, not a defect
(`docs/data_availability.md` §2).
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl

from fxvrp.data.http import Transport, get_with_retries
from fxvrp.log import get_logger, log_filter

logger = get_logger("data.fred")

_MISSING_MARKER = "."


def parse_fred_csv(text: str, series_id: str) -> pl.DataFrame:
    """Parse a fredgraph CSV into (date, value); missing markers become nulls.

    Null rows are *kept* — dropping a missing observation is a modelling decision
    that belongs to the consumer, with logging.
    """
    frame = pl.read_csv(
        io.StringIO(text),
        schema_overrides={"observation_date": pl.Date(), series_id: pl.Utf8()},
    )
    if frame.columns != ["observation_date", series_id]:
        raise ValueError(f"unexpected fredgraph columns for {series_id}: {frame.columns}")
    return frame.select(
        pl.col("observation_date").alias("date"),
        pl.when(pl.col(series_id) == _MISSING_MARKER)
        .then(None)
        .otherwise(pl.col(series_id))
        .cast(pl.Float64())
        .alias("value"),
    )


def fetch_series(
    transport: Transport,
    base_url: str,
    series_id: str,
    *,
    timeout: float,
    max_retries: int = 2,
    backoff_s: float = 1.0,
) -> pl.DataFrame:
    """Download and parse one FRED series."""
    payload = get_with_retries(
        transport,
        f"{base_url}?id={series_id}",
        timeout=timeout,
        max_retries=max_retries,
        backoff_s=backoff_s,
    )
    frame = parse_fred_csv(payload.decode("utf-8"), series_id)
    n_null = int(frame["value"].null_count())
    logger.info(
        "fetched %s: %d rows (%s .. %s), %d missing",
        series_id,
        frame.height,
        frame["date"].min(),
        frame["date"].max(),
        n_null,
    )
    return frame


def save_series(frame: pl.DataFrame, series_id: str, raw_dir: Path) -> Path:
    out_path = raw_dir / "fred" / f"{series_id}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    return out_path


def drop_missing(frame: pl.DataFrame, series_id: str) -> pl.DataFrame:
    """Remove null observations, with the mandatory filter log."""
    out = frame.drop_nulls("value")
    log_filter(logger, f"{series_id}.drop_missing", frame.height, out.height, "value is null")
    return out
