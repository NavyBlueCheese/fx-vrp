"""Dukascopy tick ingestion: raw ``.bi5`` hour files → per-day parquet.

Feed layout (verified in Phase 0 reconnaissance, `docs/data_availability.md` §3):
    {base_url}/{INSTRUMENT}/{yyyy}/{MM}/{dd}/{HH}h_ticks.bi5
where **the month is zero-indexed** (January = "00"). Each file is an
LZMA-compressed array of 20-byte big-endian records

    >IIIff  =  (ms offset within the hour, ask, bid, ask volume, bid volume)

with integer prices equal to price / price_scale (1e-5 for EURUSD). Volumes are
indicative only and are discarded (FX volume is meaningless off-exchange). An empty
body (HTTP 200, 0 bytes) is a valid "no ticks this hour" answer — weekends and
holidays — not an error.

Ingestion is resumable: a day whose parquet already exists is skipped, so the
multi-week full-history pull can be interrupted and rerun freely.
"""

from __future__ import annotations

import lzma
import struct
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import polars as pl

from fxvrp.config import DukascopyConfig
from fxvrp.data.http import Transport, get_with_retries
from fxvrp.log import get_logger

logger = get_logger("data.dukascopy")

# .bi5 record layout: constants of the wire format, not of the world
_TICK_STRUCT = struct.Struct(">IIIff")
_TICK_BYTES = _TICK_STRUCT.size
_HOURS_PER_DAY = 24

TICK_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "bid": pl.Float64(),
    "ask": pl.Float64(),
}


def hour_url(base_url: str, instrument: str, day: date, hour: int) -> str:
    """URL of one hour file. Dukascopy months are zero-indexed."""
    return (
        f"{base_url}/{instrument.upper()}/{day.year:04d}/{day.month - 1:02d}/"
        f"{day.day:02d}/{hour:02d}h_ticks.bi5"
    )


def decode_bi5(payload: bytes, hour_start: datetime, price_scale: float) -> pl.DataFrame:
    """Decode one hour file into (ts, bid, ask). Pure; empty payload → empty frame."""
    if hour_start.tzinfo is None:
        raise ValueError("hour_start must be timezone-aware UTC")
    if not payload:
        return pl.DataFrame(schema=TICK_SCHEMA)

    raw = lzma.decompress(payload)
    if len(raw) % _TICK_BYTES != 0:
        raise ValueError(f"corrupt .bi5 body: {len(raw)} bytes is not a multiple of {_TICK_BYTES}")

    n = len(raw) // _TICK_BYTES
    base_ms = int(hour_start.timestamp() * 1000)
    ts = [0] * n
    bid = [0.0] * n
    ask = [0.0] * n
    for i, (ms, ask_raw, bid_raw, _ask_vol, _bid_vol) in enumerate(
        _TICK_STRUCT.iter_unpack(raw)
    ):
        ts[i] = base_ms + ms
        ask[i] = ask_raw * price_scale
        bid[i] = bid_raw * price_scale

    return pl.DataFrame(
        {"ts": ts, "bid": bid, "ask": ask},
        schema_overrides={"ts": pl.Int64()},
    ).with_columns(
        pl.from_epoch("ts", time_unit="ms")
        .dt.replace_time_zone("UTC")
        .cast(TICK_SCHEMA["ts"])
        .alias("ts")
    )


def day_parquet_path(raw_dir: Path, instrument: str, day: date) -> Path:
    return raw_dir / "ticks" / instrument.lower() / f"{day.isoformat()}.parquet"


def fx_days(start: date, end: date) -> Iterator[date]:
    """Calendar days with possible FX activity: Sunday (late reopen) through Friday."""
    day = start
    saturday = 5  # date.weekday(): Monday=0 .. Saturday=5, Sunday=6
    while day <= end:
        if day.weekday() != saturday:
            yield day
        day += timedelta(days=1)


@dataclass(frozen=True)
class DayIngestResult:
    day: date
    n_ticks: int
    skipped: bool  # already present, not re-fetched


def ingest_day(
    transport: Transport,
    cfg: DukascopyConfig,
    day: date,
    raw_dir: Path,
) -> DayIngestResult:
    """Fetch, decode and persist one day of ticks. Skips days already on disk."""
    out_path = day_parquet_path(raw_dir, cfg.instrument, day)
    if out_path.exists():
        return DayIngestResult(day=day, n_ticks=-1, skipped=True)

    frames: list[pl.DataFrame] = []
    for hour in range(_HOURS_PER_DAY):
        url = hour_url(cfg.base_url, cfg.instrument, day, hour)
        payload = get_with_retries(
            transport,
            url,
            timeout=cfg.request_timeout_s,
            max_retries=cfg.max_retries,
            backoff_s=cfg.retry_backoff_s,
        )
        hour_start = datetime(day.year, day.month, day.day, hour, tzinfo=UTC)
        frames.append(decode_bi5(payload, hour_start, cfg.price_scale))
        if cfg.throttle_s > 0.0:
            time.sleep(cfg.throttle_s)

    day_frame = pl.concat(frames).sort("ts")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    day_frame.write_parquet(out_path)
    logger.info("ingested %s: %d ticks -> %s", day, day_frame.height, out_path)
    return DayIngestResult(day=day, n_ticks=day_frame.height, skipped=False)
