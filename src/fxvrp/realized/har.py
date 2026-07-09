"""HAR-RV forecasting (Corsi 2009) with a strict walk-forward discipline.

Model (Corsi 2009, *J. Financial Econometrics* 7(2), eq. (5), estimated in
logs): with RVÌ„_{t,h} the h-day trailing mean of daily RV,

    log RVÌ„_{t+1..t+H} = Î²â‚€ + Î²_d log RVÌ„_{t,1} + Î²_w log RVÌ„_{t,5}
                        + Î²_m log RVÌ„_{t,22} + Îµ.

Two deliberate choices, both from the brief:

1. **Direct multi-horizon regression**: the left side is the log of the
   *cumulative forward window* (our 30-calendar-day VRP window), regressed
   directly on time-t components â€” not iterated one-step forecasts.
2. **Logs with retransformation**: levels of RV are wildly right-skewed; the
   regression runs in logs and the level forecast applies the log-normal
   retransformation exp(ÏƒÌ‚Â²/2) with ÏƒÌ‚Â² the training residual variance
   (Granger & Newbold 1976 correction under Gaussian errors).

HAR-RV-CJ (Andersen, Bollerslev & Diebold 2007, Rev. Econ. Statistics 89(4)):
the daily regressor splits into a continuous part C_t and a jump part J_t via
the BNS test at level ``jump_alpha``:
    C_t = RV_t if z_t <= z_crit else BPV_t,   J_t = (RV_t - BPV_t)Â·1{z_t > z_crit},
with regressors log CÌ„ (three horizons) and log(1 + J_t).

Walk-forward: expanding window, refit every ``refit_every_days`` rows, and the
training slice passes through the lookahead guard at every refit â€” a
contaminated information set crashes rather than flatters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl
from scipy import stats

from fxvrp._types import FloatArray
from fxvrp.config import VrpConfig
from fxvrp.log import get_logger
from fxvrp.lookahead import assert_information_set

logger = get_logger("realized.har")

_MIN_ROWS_PER_PARAM = 3  # refuse to fit with fewer than 3 observations per coefficient


@dataclass(frozen=True)
class HARFit:
    coefficients: FloatArray  # intercept first, then one per feature column
    residual_variance: float  # in log space, for the retransformation
    n_obs: int
    feature_names: tuple[str, ...]


def continuous_jump_split(panel: pl.DataFrame, jump_alpha: float) -> pl.DataFrame:
    """ABD (2007) C/J decomposition of daily RV using the BNS statistic."""
    z_crit = float(stats.norm.ppf(1.0 - jump_alpha))
    return panel.with_columns(
        pl.when(pl.col("bns_z") > z_crit)
        .then(pl.col("bpv"))
        .otherwise(pl.col("rv"))
        .alias("c_part"),
        pl.when(pl.col("bns_z") > z_crit)
        .then(pl.max_horizontal(pl.col("rv") - pl.col("bpv"), pl.lit(0.0)))
        .otherwise(0.0)
        .alias("j_part"),
    )


def har_features(
    panel: pl.DataFrame,
    lags: tuple[int, ...],
    *,
    rv_col: str = "rv_total",
    jump_split: bool = False,
) -> pl.DataFrame:
    """Attach log HAR regressors at each date, using information up to t only.

    Trailing means are over panel *rows* (trading days), inclusive of t.
    """
    base_col = "c_part" if jump_split else rv_col
    frame = panel.sort("day")
    exprs = [
        pl.col(base_col)
        .rolling_mean(window_size=lag, min_samples=lag)
        .log()
        .alias(f"log_rv_{lag}d")
        for lag in lags
    ]
    if jump_split:
        exprs.append((pl.col("j_part") + 1.0).log().alias("log_1p_jump"))
    return frame.with_columns(exprs)


def feature_columns(lags: tuple[int, ...], jump_split: bool) -> tuple[str, ...]:
    cols = tuple(f"log_rv_{lag}d" for lag in lags)
    return (*cols, "log_1p_jump") if jump_split else cols


def forward_realized_window(
    panel: pl.DataFrame,
    cfg: VrpConfig,
    *,
    rv_col: str = "rv_total",
) -> pl.DataFrame:
    """Attach the forward 30-calendar-day cumulative RV (annualised) at each date.

    The window (t, t + forward_window_calendar_days] sums daily RV over the FX
    days present; windows covering less than ``min_window_coverage`` of their
    expected weekday count are null (an ingestion gap must not fake low
    variance). This column is the *ex-post* object: descriptive only, never a
    feature.
    """
    frame = panel.sort("day")
    days = frame["day"].to_list()
    values = frame[rv_col].to_list()
    horizon = cfg.forward_window_calendar_days
    ann = float(cfg.annualize_days) / horizon

    n = len(days)
    fwd: list[float | None] = []
    coverage: list[float] = []
    for i in range(n):
        window_end = days[i] + timedelta(days=horizon)
        total = 0.0
        count = 0
        has_null = False
        k = i + 1
        while k < n and days[k] <= window_end:
            value = values[k]
            if value is None:
                has_null = True
            else:
                total += float(value)
            count += 1
            k += 1
        expected = _expected_weekdays(days[i], window_end)
        cov = count / expected if expected else 0.0
        usable = (not has_null) and cov >= cfg.min_window_coverage
        fwd.append(total * ann if usable else None)
        coverage.append(cov)

    result = pl.DataFrame(
        {"day": days, "rv_fwd_ann": fwd, "window_coverage": coverage},
        schema_overrides={"rv_fwd_ann": pl.Float64()},
    )
    return frame.join(result, on="day", how="left")


def _expected_weekdays(start_exclusive: date, end_inclusive: date) -> int:
    day = start_exclusive + timedelta(days=1)
    count = 0
    saturday = 5
    while day <= end_inclusive:
        if day.weekday() < saturday:
            count += 1
        day += timedelta(days=1)
    return count


def fit_har(
    train: pl.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
) -> HARFit:
    """OLS in log space on complete rows."""
    complete = train.drop_nulls([*feature_cols, target_col])
    n_params = len(feature_cols) + 1
    if complete.height < n_params * _MIN_ROWS_PER_PARAM:
        raise ValueError(f"only {complete.height} complete rows to fit {n_params} parameters")
    x = np.column_stack(
        [np.ones(complete.height)]
        + [complete[col].to_numpy().astype(np.float64) for col in feature_cols]
    )
    y = np.log(complete[target_col].to_numpy().astype(np.float64))
    coefs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefs
    dof = max(complete.height - n_params, 1)
    return HARFit(
        coefficients=coefs.astype(np.float64),
        residual_variance=float(residuals @ residuals / dof),
        n_obs=complete.height,
        feature_names=feature_cols,
    )


def predict_har(fit: HARFit, rows: pl.DataFrame) -> FloatArray:
    """Level forecast with the log-normal retransformation exp(ÏƒÌ‚Â²/2)."""
    x = np.column_stack(
        [np.ones(rows.height)]
        + [rows[col].to_numpy().astype(np.float64) for col in fit.feature_names]
    )
    log_forecast = x @ fit.coefficients
    return np.asarray(np.exp(log_forecast + fit.residual_variance / 2.0), dtype=np.float64)


def walk_forward_forecast(
    panel_with_features: pl.DataFrame,
    cfg: VrpConfig,
    *,
    feature_cols: tuple[str, ...],
    target_col: str = "rv_fwd_ann",
) -> pl.DataFrame:
    """Expanding-window walk-forward: (day, rv_fwd_hat_ann), guarded at every refit.

    The training set at decision date t contains only rows whose *target
    window has closed* by t (day <= t - horizon), so the target itself never
    leaks future information into the fit; the lookahead guard then re-checks
    the timestamps mechanically.
    """
    frame = panel_with_features.sort("day")
    days = frame["day"].to_list()
    horizon = timedelta(days=cfg.forward_window_calendar_days)

    forecasts: list[float | None] = [None] * len(days)
    current_fit: HARFit | None = None
    last_refit_index = -(10**9)

    for i, day_t in enumerate(days):
        train = frame.filter(pl.col("day") <= day_t - horizon)
        if train.height >= cfg.min_train_days and (
            current_fit is None or i - last_refit_index >= cfg.refit_every_days
        ):
            assert_information_set(train, asof=day_t, ts_col="day")
            try:
                current_fit = fit_har(train, feature_cols, target_col)
                last_refit_index = i
            except ValueError as error:
                logger.info("refit skipped at %s: %s", day_t, error)
        if current_fit is None:
            continue
        row = frame[i].select(list(feature_cols))
        if any(row[col].null_count() for col in feature_cols):
            continue
        forecasts[i] = float(predict_har(current_fit, row)[0])

    return pl.DataFrame({"day": days, "rv_fwd_hat_ann": forecasts})
