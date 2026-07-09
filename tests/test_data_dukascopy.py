from __future__ import annotations

import lzma
import struct
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from fxvrp.config import Config, DukascopyConfig
from fxvrp.data.dukascopy import (
    day_parquet_path,
    decode_bi5,
    fx_days,
    hour_url,
    ingest_day,
)

PRICE_SCALE = 1e-5


def _encode_ticks(ticks: list[tuple[int, int, int]]) -> bytes:
    """Build a .bi5 body from (ms, ask_raw, bid_raw) records."""
    raw = b"".join(struct.pack(">IIIff", ms, ask, bid, 1.0, 1.0) for ms, ask, bid in ticks)
    return lzma.compress(raw)


def test_hour_url_month_is_zero_indexed() -> None:
    url = hour_url("https://x", "EURUSD", date(2020, 1, 6), 7)
    assert url == "https://x/EURUSD/2020/00/06/07h_ticks.bi5"
    url_dec = hour_url("https://x", "eurusd", date(2019, 12, 31), 23)
    assert "/2019/11/31/23h_ticks.bi5" in url_dec
    assert "/EURUSD/" in url_dec  # instrument upper-cased


def test_decode_bi5_roundtrip() -> None:
    hour_start = datetime(2020, 3, 9, 10, tzinfo=UTC)
    payload = _encode_ticks([(627, 113880, 113876), (3425, 113879, 113876)])
    frame = decode_bi5(payload, hour_start, PRICE_SCALE)
    assert frame.height == 2
    assert frame["ask"].to_list() == pytest.approx([1.13880, 1.13879])
    assert frame["bid"].to_list() == pytest.approx([1.13876, 1.13876])
    first_ts = frame["ts"][0]
    assert first_ts == datetime(2020, 3, 9, 10, 0, 0, 627_000, tzinfo=UTC)


def test_decode_bi5_empty_payload_is_empty_frame() -> None:
    frame = decode_bi5(b"", datetime(2020, 3, 8, 0, tzinfo=UTC), PRICE_SCALE)
    assert frame.height == 0
    assert frame.columns == ["ts", "bid", "ask"]


def test_decode_bi5_rejects_naive_hour_start() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        decode_bi5(b"", datetime(2020, 3, 8, 0), PRICE_SCALE)


def test_decode_bi5_rejects_corrupt_body() -> None:
    corrupt = lzma.compress(b"\x00" * 19)
    with pytest.raises(ValueError, match="corrupt"):
        decode_bi5(corrupt, datetime(2020, 3, 9, 0, tzinfo=UTC), PRICE_SCALE)


def test_fx_days_excludes_saturdays_only() -> None:
    days = list(fx_days(date(2020, 3, 6), date(2020, 3, 10)))  # Fri..Tue
    assert date(2020, 3, 7) not in days  # Saturday
    assert date(2020, 3, 8) in days  # Sunday (late reopen)
    assert len(days) == 4


@dataclass
class _StubResponse:
    payload: bytes
    status_code: int = 200

    @property
    def content(self) -> bytes:
        return self.payload

    @property
    def text(self) -> str:
        return self.payload.decode("utf-8", errors="replace")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@dataclass
class _StubTransport:
    responses: dict[str, bytes]
    calls: list[str] = field(default_factory=list)

    def get(self, url: str, timeout: float) -> _StubResponse:
        self.calls.append(url)
        return _StubResponse(self.responses.get(url, b""))


def _test_cfg(config: Config) -> DukascopyConfig:
    base = config.dukascopy
    return DukascopyConfig(
        base_url=base.base_url,
        instrument=base.instrument,
        price_scale=base.price_scale,
        pip=base.pip,
        start_date=base.start_date,
        end_date=base.end_date,
        sample_start_date=base.sample_start_date,
        sample_end_date=base.sample_end_date,
        max_retries=0,
        retry_backoff_s=0.0,
        request_timeout_s=1.0,
        throttle_s=0.0,
    )


def test_ingest_day_writes_sorted_parquet_and_is_resumable(tmp_path: Path, config: Config) -> None:
    cfg = _test_cfg(config)
    day = date(2020, 3, 9)
    hour_10 = hour_url(cfg.base_url, cfg.instrument, day, 10)
    hour_11 = hour_url(cfg.base_url, cfg.instrument, day, 11)
    transport = _StubTransport(
        {
            hour_10: _encode_ticks([(500, 113880, 113876)]),
            hour_11: _encode_ticks([(100, 113990, 113985)]),
        }
    )

    result = ingest_day(transport, cfg, day, tmp_path)
    assert not result.skipped
    assert result.n_ticks == 2
    assert len(transport.calls) == 24  # all hours attempted

    written = pl.read_parquet(day_parquet_path(tmp_path, cfg.instrument, day))
    assert written["ts"].is_sorted()
    assert written.height == 2

    # second run must not re-fetch
    transport.calls.clear()
    again = ingest_day(transport, cfg, day, tmp_path)
    assert again.skipped
    assert transport.calls == []
