"""€STR from the ECB Data Portal SDMX CSV API.

Series ``EST/B.EU000A2X2A25.WT`` (volume-weighted trimmed mean rate), daily from
2019-10-01. The EUR overnight history is EONIA (FRED ``EONIARATE``) until
2019-09-30 and €STR thereafter; EONIA was redefined as €STR + 8.5 bp in Oct 2019,
and per `docs/conventions.md` rule 11 the splice keeps both series as published —
no back-adjustment.
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl

from fxvrp.config import EcbConfig
from fxvrp.data.http import Transport, get_with_retries
from fxvrp.log import get_logger

logger = get_logger("data.ecb")


def parse_estr_csv(text: str) -> pl.DataFrame:
    """Extract (date, rate) from the SDMX csvdata payload. Pure."""
    frame = pl.read_csv(io.StringIO(text), schema_overrides={"TIME_PERIOD": pl.Date()})
    required = {"TIME_PERIOD", "OBS_VALUE"}
    if not required.issubset(frame.columns):
        raise ValueError(f"unexpected ECB csvdata columns: {frame.columns}")
    return frame.select(
        pl.col("TIME_PERIOD").alias("date"),
        pl.col("OBS_VALUE").cast(pl.Float64()).alias("rate"),
    ).sort("date")


def fetch_estr(
    transport: Transport,
    cfg: EcbConfig,
    *,
    timeout: float,
    max_retries: int = 2,
    backoff_s: float = 1.0,
) -> pl.DataFrame:
    """Download the full €STR history from the configured start."""
    url = (
        f"{cfg.base_url}/{cfg.estr_series}?format=csvdata&startPeriod={cfg.estr_start.isoformat()}"
    )
    payload = get_with_retries(
        transport, url, timeout=timeout, max_retries=max_retries, backoff_s=backoff_s
    )
    frame = parse_estr_csv(payload.decode("utf-8"))
    logger.info(
        "fetched ESTR: %d rows (%s .. %s)", frame.height, frame["date"].min(), frame["date"].max()
    )
    return frame


def save_estr(frame: pl.DataFrame, raw_dir: Path) -> Path:
    out_path = raw_dir / "ecb" / "estr.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    return out_path
