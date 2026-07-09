"""Stylized-facts machinery: moments, and HAC inference that respects overlap."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fxvrp.analysis.stylized_facts import facts_table, series_facts


def test_iid_normal_series_facts() -> None:
    rng = np.random.default_rng(4)
    frame = pl.DataFrame({"x": rng.normal(0.0, 1.0, 20_000)})
    facts = series_facts(frame, "x", hac_maxlags=10)
    assert facts.n_obs == 20_000
    assert abs(facts.mean) < 0.03
    assert abs(facts.hac_t_stat) < 3.0
    assert facts.std == pytest.approx(1.0, rel=0.03)
    assert abs(facts.skewness) < 0.06
    assert abs(facts.excess_kurtosis) < 0.12
    assert facts.fraction_positive == pytest.approx(0.5, abs=0.02)
    assert abs(facts.autocorr_1d) < 0.03


def test_hac_deflates_t_stat_on_overlapping_series() -> None:
    """Overlap-induced autocorrelation must shrink the t-statistic (brief §7)."""
    rng = np.random.default_rng(7)
    innovations = rng.normal(0.1, 1.0, 3_000)
    overlapping = np.convolve(innovations, np.ones(22) / 22, mode="valid")
    frame = pl.DataFrame({"x": overlapping})

    hac = series_facts(frame, "x", hac_maxlags=44)
    naive = series_facts(frame, "x", hac_maxlags=0)
    assert hac.autocorr_1d > 0.9  # the overlap creates massive autocorrelation
    assert abs(hac.hac_t_stat) < abs(naive.hac_t_stat) / 2.5


def test_facts_table_skips_empty_columns() -> None:
    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [None, None, None]}).with_columns(
        pl.col("b").cast(pl.Float64())
    )
    table = facts_table(frame, ["a", "b"], hac_maxlags=1)
    assert table["series"].to_list() == ["a"]


def test_empty_series_raises() -> None:
    frame = pl.DataFrame({"x": []}, schema={"x": pl.Float64()})
    with pytest.raises(ValueError, match="no observations"):
        series_facts(frame, "x", hac_maxlags=1)
