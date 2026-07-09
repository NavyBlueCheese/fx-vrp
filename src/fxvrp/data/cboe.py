"""CBOE delayed-quotes option chains: the forward-collected FXE surface panel.

Endpoint (verified live in Phase 0): ``{base}/{SYMBOL}.json`` with index symbols
prefixed by underscore (``_SPX``). The payload carries the underlying spot
(``data.current_price``), a quote timestamp in US/Eastern (``timestamp``) and one
record per contract with two-sided quotes, IV, greeks, open interest and volume —
strictly richer than any free historical source found in reconnaissance.

Contract names follow OCC symbology: ``{root}{yymmdd}{C|P}{strike*1000, 8 digits}``.
The root may carry adjustment suffixes (e.g. ``FXE1``), so parsing anchors on the
fixed-width 15-character tail rather than on the root.

EVZ was decommissioned on 2025-03-11, so this scraper is the only continuation of
the FXE implied-variance series; ``_SPX`` is collected alongside to validate the
MFIV implementation against published VIX (ADR 0002).
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from fxvrp.data.http import Transport, get_with_retries
from fxvrp.log import get_logger

logger = get_logger("data.cboe")

_EASTERN = ZoneInfo("America/New_York")
# OCC symbology constants (format facts): 6-digit date, C/P flag, strike in mills
_OCC_TAIL = re.compile(r"^(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
_OCC_TAIL_LEN = 15
_OCC_STRIKE_DIVISOR = 1000.0
_OCC_CENTURY = 2000


@dataclass(frozen=True)
class OccContract:
    root: str
    expiry: date
    call_put: str  # "C" or "P"
    strike: float


def parse_occ_symbol(symbol: str) -> OccContract:
    """Split an OCC option symbol into its components. Pure."""
    compact = symbol.replace(" ", "")
    if len(compact) <= _OCC_TAIL_LEN:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, tail = compact[:-_OCC_TAIL_LEN], compact[-_OCC_TAIL_LEN:]
    match = _OCC_TAIL.match(tail)
    if match is None:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    ymd = match["ymd"]
    expiry = date(_OCC_CENTURY + int(ymd[0:2]), int(ymd[2:4]), int(ymd[4:6]))
    return OccContract(
        root=root,
        expiry=expiry,
        call_put=match["cp"],
        strike=int(match["strike"]) / _OCC_STRIKE_DIVISOR,
    )


def _opt_float(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _opt_int(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def parse_quote_timestamp(raw: str) -> datetime:
    """CBOE timestamps are naive US/Eastern; return an aware UTC datetime."""
    naive = datetime.fromisoformat(raw)
    return naive.replace(tzinfo=_EASTERN).astimezone(UTC)


def parse_chain(
    payload: Mapping[str, Any],
    symbol: str,
    fetched_at: datetime,
) -> pl.DataFrame:
    """Normalize one delayed-quotes payload into a flat contract table. Pure."""
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware UTC")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"malformed CBOE payload for {symbol}: no 'data' object")
    options = data.get("options")
    if not isinstance(options, list):
        raise ValueError(f"malformed CBOE payload for {symbol}: no 'options' list")

    spot = _opt_float(data, "current_price")
    raw_ts = payload.get("timestamp")
    quote_time = parse_quote_timestamp(raw_ts) if isinstance(raw_ts, str) else None

    records: list[dict[str, Any]] = []
    n_unparsed = 0
    for entry in options:
        if not isinstance(entry, dict):
            n_unparsed += 1
            continue
        name = entry.get("option")
        if not isinstance(name, str):
            n_unparsed += 1
            continue
        try:
            contract = parse_occ_symbol(name)
        except ValueError:
            n_unparsed += 1
            continue
        records.append(
            {
                "underlying": symbol,
                "contract": name,
                "root": contract.root,
                "expiry": contract.expiry,
                "call_put": contract.call_put,
                "strike": contract.strike,
                "bid": _opt_float(entry, "bid"),
                "ask": _opt_float(entry, "ask"),
                "bid_size": _opt_int(entry, "bid_size"),
                "ask_size": _opt_int(entry, "ask_size"),
                "iv": _opt_float(entry, "iv"),
                "delta": _opt_float(entry, "delta"),
                "gamma": _opt_float(entry, "gamma"),
                "theta": _opt_float(entry, "theta"),
                "vega": _opt_float(entry, "vega"),
                "rho": _opt_float(entry, "rho"),
                "open_interest": _opt_int(entry, "open_interest"),
                "volume": _opt_int(entry, "volume"),
                "last_trade_price": _opt_float(entry, "last_trade_price"),
                "theo": _opt_float(entry, "theo"),
                "spot": spot,
                "quote_time": quote_time,
                "fetched_at": fetched_at,
            }
        )
    if n_unparsed:
        logger.warning("%s: %d option records could not be parsed", symbol, n_unparsed)

    return pl.DataFrame(
        records,
        schema_overrides={
            "expiry": pl.Date(),
            "quote_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            "fetched_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


def fetch_chain(
    transport: Transport,
    base_url: str,
    symbol: str,
    *,
    timeout: float,
    max_retries: int = 2,
    backoff_s: float = 1.0,
) -> dict[str, Any]:
    """Download one chain payload as parsed JSON."""
    payload = get_with_retries(
        transport,
        f"{base_url}/{symbol}.json",
        timeout=timeout,
        max_retries=max_retries,
        backoff_s=backoff_s,
    )
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"CBOE payload for {symbol} is not a JSON object")
    return parsed


def snapshot_paths(raw_dir: Path, symbol: str, snapshot_date: date) -> tuple[Path, Path]:
    """(raw json.gz, normalized parquet) locations for one (symbol, date) snapshot."""
    stem = snapshot_date.isoformat()
    base = raw_dir / "chains" / symbol.lstrip("_").lower()
    return base / f"{stem}.json.gz", base / f"{stem}.parquet"


def save_snapshot(
    payload: Mapping[str, Any],
    frame: pl.DataFrame,
    raw_dir: Path,
    symbol: str,
    snapshot_date: date,
) -> tuple[Path, Path]:
    """Persist the immutable raw payload and its normalized table."""
    json_path, parquet_path = snapshot_paths(raw_dir, symbol, snapshot_date)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(json_path, "wt", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, separators=(",", ":"))
    frame.write_parquet(parquet_path)
    return json_path, parquet_path
