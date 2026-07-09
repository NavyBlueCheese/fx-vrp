from __future__ import annotations

from datetime import date

import pytest

from fxvrp.data.ecb import parse_estr_csv
from fxvrp.data.fred import drop_missing, parse_fred_csv

FRED_FIXTURE = """observation_date,EVZCLS
2007-11-01,8.09
2007-11-02,8.12
2007-11-05,.
2025-03-11,10.68
"""

ECB_FIXTURE = (
    "KEY,FREQ,BENCHMARK_ITEM,DATA_TYPE_EST,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
    "EST.B.EU000A2X2A25.WT,B,EU000A2X2A25,WT,2019-10-01,-0.549,A\n"
    "EST.B.EU000A2X2A25.WT,B,EU000A2X2A25,WT,2019-10-02,-0.551,A\n"
)


def test_parse_fred_csv_keeps_missing_as_null() -> None:
    frame = parse_fred_csv(FRED_FIXTURE, "EVZCLS")
    assert frame.columns == ["date", "value"]
    assert frame.height == 4
    assert frame["value"].null_count() == 1
    assert frame["date"][0] == date(2007, 11, 1)
    assert frame["value"][0] == pytest.approx(8.09)
    assert frame["value"][3] == pytest.approx(10.68)


def test_drop_missing_removes_only_nulls() -> None:
    frame = parse_fred_csv(FRED_FIXTURE, "EVZCLS")
    cleaned = drop_missing(frame, "EVZCLS")
    assert cleaned.height == 3
    assert cleaned["value"].null_count() == 0


def test_parse_fred_csv_rejects_wrong_series() -> None:
    with pytest.raises(ValueError, match="unexpected fredgraph columns"):
        parse_fred_csv(FRED_FIXTURE, "VIXCLS")


def test_parse_estr_csv_extracts_date_rate() -> None:
    frame = parse_estr_csv(ECB_FIXTURE)
    assert frame.columns == ["date", "rate"]
    assert frame.height == 2
    assert frame["date"][0] == date(2019, 10, 1)
    assert frame["rate"][0] == pytest.approx(-0.549)
    assert frame["date"].is_sorted()


def test_parse_estr_csv_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match="unexpected ECB csvdata columns"):
        parse_estr_csv("A,B\n1,2\n")
