from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from fxvrp.data.cboe import parse_chain, parse_occ_symbol, parse_quote_timestamp

FETCHED_AT = datetime(2026, 7, 8, 20, 30, tzinfo=UTC)


def _payload() -> dict[str, object]:
    return {
        "timestamp": "2026-07-08 16:00:00",
        "data": {
            "current_price": 105.307,
            "options": [
                {
                    "option": "FXE260717C00105000",
                    "bid": 1.05,
                    "ask": 1.20,
                    "bid_size": 10,
                    "ask_size": 12,
                    "iv": 0.081,
                    "delta": 0.52,
                    "gamma": 0.15,
                    "theta": -0.01,
                    "vega": 0.12,
                    "rho": 0.05,
                    "open_interest": 321,
                    "volume": 12,
                    "last_trade_price": 1.10,
                    "theo": 1.12,
                },
                {
                    "option": "FXE260717P00104000",
                    "bid": 0.0,
                    "ask": 0.55,
                    "iv": 0.083,
                    "open_interest": 15,
                    "volume": 0,
                },
                {"option": "GARBAGE"},
            ],
        },
    }


def test_parse_occ_symbol_plain_and_index_roots() -> None:
    fxe = parse_occ_symbol("FXE260717C00105000")
    assert fxe.root == "FXE"
    assert fxe.expiry == date(2026, 7, 17)
    assert fxe.call_put == "C"
    assert fxe.strike == pytest.approx(105.0)

    spxw = parse_occ_symbol("SPXW261218P05000000")
    assert spxw.root == "SPXW"
    assert spxw.strike == pytest.approx(5000.0)
    assert spxw.expiry == date(2026, 12, 18)

    adjusted = parse_occ_symbol("FXE1260717C00105500")
    assert adjusted.root == "FXE1"
    assert adjusted.strike == pytest.approx(105.5)


def test_parse_occ_symbol_rejects_garbage() -> None:
    for bad in ("GARBAGE", "FXE2607X7C00105000", ""):
        with pytest.raises(ValueError):
            parse_occ_symbol(bad)


def test_parse_quote_timestamp_is_eastern_to_utc() -> None:
    utc = parse_quote_timestamp("2026-07-08 16:00:00")
    assert utc == datetime(2026, 7, 8, 20, 0, tzinfo=UTC)  # EDT = UTC-4 in July
    winter = parse_quote_timestamp("2026-01-08 16:00:00")
    assert winter == datetime(2026, 1, 8, 21, 0, tzinfo=UTC)  # EST = UTC-5


def test_parse_chain_normalizes_and_skips_unparsable() -> None:
    frame = parse_chain(_payload(), "FXE", FETCHED_AT)
    assert frame.height == 2  # GARBAGE row skipped, warned
    assert set(frame["call_put"].to_list()) == {"C", "P"}
    assert frame["spot"].to_list() == pytest.approx([105.307, 105.307])
    assert frame["quote_time"][0] == datetime(2026, 7, 8, 20, 0, tzinfo=UTC)
    assert frame["fetched_at"][0] == FETCHED_AT
    call = frame.filter(frame["call_put"] == "C")
    assert call["open_interest"][0] == 321
    assert call["iv"][0] == pytest.approx(0.081)


def test_parse_chain_rejects_naive_fetched_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_chain(_payload(), "FXE", datetime(2026, 7, 8, 20, 30))


def test_parse_chain_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match="no 'data'"):
        parse_chain({"nope": 1}, "FXE", FETCHED_AT)
    with pytest.raises(ValueError, match="no 'options'"):
        parse_chain({"data": {"current_price": 1.0}}, "FXE", FETCHED_AT)
