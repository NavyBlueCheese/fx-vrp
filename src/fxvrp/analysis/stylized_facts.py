"""Stylized facts of the VRP series, with autocorrelation-honest inference.

The 30-day forward window overlaps across days, so the ex-post VRP is severely
autocorrelated by construction; the mean's t-statistic uses Newey-West (1987)
HAC standard errors with a bandwidth covering the overlap. OLS standard errors
on overlapping windows would make everything significant and all of it false
(brief §7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm

from fxvrp._types import FloatArray


@dataclass(frozen=True)
class SeriesFacts:
    n_obs: int
    mean: float
    hac_t_stat: float  # for H0: mean = 0, Newey-West
    std: float
    skewness: float
    excess_kurtosis: float
    fraction_positive: float
    autocorr_1d: float
    autocorr_22d: float


def _autocorr(values: FloatArray, lag: int) -> float:
    if values.size <= lag + 1:
        return float("nan")
    a = values[:-lag] - values[:-lag].mean()
    b = values[lag:] - values[lag:].mean()
    denom = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b / denom) if denom > 0.0 else float("nan")


def series_facts(frame: pl.DataFrame, col: str, hac_maxlags: int) -> SeriesFacts:
    """Distributional and dependence summary of one series column."""
    values = frame.drop_nulls(col)[col].to_numpy().astype(np.float64)
    if values.size == 0:
        raise ValueError(f"no observations in column {col!r}")

    ols = sm.OLS(values, np.ones_like(values)).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
    )
    centered = values - values.mean()
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    m2 = float(np.mean(centered**2))
    skew = float(np.mean(centered**3) / m2**1.5) if m2 > 0 else float("nan")
    kurt = float(np.mean(centered**4) / m2**2 - 3.0) if m2 > 0 else float("nan")
    return SeriesFacts(
        n_obs=int(values.size),
        mean=float(values.mean()),
        hac_t_stat=float(ols.tvalues[0]),
        std=std,
        skewness=skew,
        excess_kurtosis=kurt,
        fraction_positive=float(np.mean(values > 0.0)),
        autocorr_1d=_autocorr(values, 1),
        autocorr_22d=_autocorr(values, 22),
    )


def facts_table(frame: pl.DataFrame, columns: list[str], hac_maxlags: int) -> pl.DataFrame:
    """Stack ``series_facts`` for several columns into a report-ready frame."""
    rows = []
    for col in columns:
        available = frame.drop_nulls(col)
        if available.height == 0:
            continue
        facts = series_facts(frame, col, hac_maxlags)
        rows.append(
            {
                "series": col,
                "n": facts.n_obs,
                "mean": facts.mean,
                "hac_t": facts.hac_t_stat,
                "std": facts.std,
                "skew": facts.skewness,
                "ex_kurt": facts.excess_kurtosis,
                "frac_pos": facts.fraction_positive,
                "ac_1d": facts.autocorr_1d,
                "ac_22d": facts.autocorr_22d,
            }
        )
    return pl.DataFrame(rows)
