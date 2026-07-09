"""HAR machinery: feature alignment, forward targets, fit recovery, and the
walk-forward purity acceptance test (forecast at t unchanged when the future
is deleted)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from fxvrp.config import Config, VrpConfig
from fxvrp.realized.har import (
    continuous_jump_split,
    feature_columns,
    fit_har,
    forward_realized_window,
    har_features,
    predict_har,
    walk_forward_forecast,
)


def _weekdays(start: date, n: int) -> list[date]:
    days = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _panel(rv: list[float], start: date = date(2018, 1, 1)) -> pl.DataFrame:
    days = _weekdays(start, len(rv))
    return pl.DataFrame(
        {
            "day": days,
            "rv_total": rv,
            "rv": rv,
            "bpv": [v * 0.9 for v in rv],
            "bns_z": [0.0] * len(rv),
        }
    )


def _vrp_cfg(config: Config, **overrides: object) -> VrpConfig:
    base = config.vrp
    values = {
        "har_lags": base.har_lags,
        "refit_every_days": base.refit_every_days,
        "min_train_days": base.min_train_days,
        "forward_window_calendar_days": base.forward_window_calendar_days,
        "min_window_coverage": base.min_window_coverage,
        "hac_maxlags": base.hac_maxlags,
        "jump_alpha": base.jump_alpha,
        "annualize_days": base.annualize_days,
    }
    values.update(overrides)
    return VrpConfig(**values)  # type: ignore[arg-type]


def test_har_features_use_only_past_information(config: Config) -> None:
    rv = [1.0] * 21 + [100.0]  # today spikes
    frame = har_features(_panel(rv), config.vrp.har_lags, rv_col="rv_total")
    last = frame.row(-1, named=True)
    prev = frame.row(-2, named=True)
    # daily component at t is log RV_t: sees the spike today, not yesterday
    assert last["log_rv_1d"] == pytest.approx(np.log(100.0))
    assert prev["log_rv_1d"] == pytest.approx(0.0)
    # weekly = log mean of the trailing 5 including today
    assert last["log_rv_5d"] == pytest.approx(np.log((4 * 1.0 + 100.0) / 5))
    # monthly window not yet full on early rows -> null
    assert frame.row(10, named=True)["log_rv_22d"] is None


def test_forward_window_sums_future_only_and_annualises(config: Config) -> None:
    cfg = _vrp_cfg(config, forward_window_calendar_days=7, min_window_coverage=0.5)
    rv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    frame = forward_realized_window(_panel(rv), cfg)
    first = frame.row(0, named=True)
    # 7 calendar days after a Monday cover Tue..Fri + Mon = 5 rows: rv 2+3+4+5+6
    assert first["rv_fwd_ann"] == pytest.approx((2 + 3 + 4 + 5 + 6) * 365 / 7)
    assert first["window_coverage"] == pytest.approx(1.0)
    # the last row has no future -> null
    assert frame.row(-1, named=True)["rv_fwd_ann"] is None


def test_forward_window_nulls_on_poor_coverage(config: Config) -> None:
    cfg = _vrp_cfg(config, forward_window_calendar_days=7, min_window_coverage=0.9)
    # drop two interior days: coverage 3/5 < 0.9 -> null
    days = _weekdays(date(2018, 1, 1), 6)
    sparse = pl.DataFrame(
        {"day": [days[0], days[1], days[3], days[5]], "rv_total": [1.0, 2.0, 4.0, 6.0]}
    )
    frame = forward_realized_window(sparse, cfg)
    assert frame.row(0, named=True)["rv_fwd_ann"] is None


def test_fit_recovers_known_coefficients(config: Config) -> None:
    rng = np.random.default_rng(3)
    n = 2_000
    x1, x2 = rng.normal(-9, 1, n), rng.normal(-9, 1, n)
    log_y = 0.5 + 0.6 * x1 + 0.3 * x2 + rng.normal(0, 0.05, n)
    frame = pl.DataFrame({"log_rv_1d": x1, "log_rv_5d": x2, "rv_fwd_ann": np.exp(log_y)})
    fit = fit_har(frame, ("log_rv_1d", "log_rv_5d"), "rv_fwd_ann")
    assert fit.coefficients == pytest.approx([0.5, 0.6, 0.3], abs=0.02)
    assert fit.residual_variance == pytest.approx(0.05**2, rel=0.15)


def test_retransformation_makes_level_forecasts_mean_unbiased(config: Config) -> None:
    """E[Y] = exp(mu + s^2/2): without the correction, level forecasts are biased low."""
    rng = np.random.default_rng(9)
    n = 40_000
    x = rng.normal(-9, 0.5, n)
    sigma = 0.8  # large log-noise so the correction is material (~38%)
    y = np.exp(1.0 + 1.0 * x + rng.normal(0, sigma, n))
    frame = pl.DataFrame({"log_rv_1d": x, "rv_fwd_ann": y})
    fit = fit_har(frame, ("log_rv_1d",), "rv_fwd_ann")
    predictions = predict_har(fit, frame)
    ratio = float(predictions.mean() / y.mean())
    assert ratio == pytest.approx(1.0, abs=0.05)
    naive_ratio = float(np.exp(np.log(predictions) - fit.residual_variance / 2.0).mean() / y.mean())
    assert naive_ratio < 0.75  # the uncorrected forecast is badly biased low


def test_continuous_jump_split(config: Config) -> None:
    panel = _panel([1.0, 1.0, 1.0]).with_columns(
        pl.Series("bns_z", [0.0, 5.0, 0.0]), pl.Series("bpv", [0.9, 0.4, 0.9])
    )
    split = continuous_jump_split(panel, jump_alpha=0.01)
    rows = split.to_dicts()
    assert rows[0]["c_part"] == 1.0 and rows[0]["j_part"] == 0.0
    assert rows[1]["c_part"] == 0.4 and rows[1]["j_part"] == pytest.approx(0.6)
    assert feature_columns((1, 5), jump_split=True)[-1] == "log_1p_jump"


@pytest.mark.slow
def test_walk_forward_purity(config: Config) -> None:
    """ACCEPTANCE: the forecast for date t is unchanged when data after t is deleted."""
    rng = np.random.default_rng(21)
    n = 900
    log_rv = np.empty(n)
    log_rv[0] = -9.0
    for i in range(1, n):  # persistent volatility in logs
        log_rv[i] = -0.45 + 0.95 * log_rv[i - 1] + rng.normal(0, 0.3)
    panel = _panel(list(np.exp(log_rv)))
    cfg = _vrp_cfg(config, min_train_days=200, refit_every_days=21)

    features = har_features(panel, cfg.har_lags, rv_col="rv_total")
    full = forward_realized_window(features, cfg)
    cols = feature_columns(cfg.har_lags, jump_split=False)
    forecast_full = walk_forward_forecast(full, cfg, feature_cols=cols)

    cutoff_row = 700
    cutoff_day = full["day"][cutoff_row]
    truncated = full.filter(pl.col("day") <= cutoff_day)
    forecast_trunc = walk_forward_forecast(truncated, cfg, feature_cols=cols)

    y_full = forecast_full.filter(pl.col("day") == cutoff_day)["rv_fwd_hat_ann"][0]
    y_trunc = forecast_trunc.filter(pl.col("day") == cutoff_day)["rv_fwd_hat_ann"][0]
    assert y_full is not None
    assert y_full == pytest.approx(y_trunc, rel=1e-12)

    # and the forecasts have skill on this persistent DGP: better than the
    # expanding-mean naive forecast in log MSE
    joined = full.join(forecast_full, on="day", how="inner").drop_nulls(
        ["rv_fwd_ann", "rv_fwd_hat_ann"]
    )
    err_har = np.log(joined["rv_fwd_hat_ann"] / joined["rv_fwd_ann"]).to_numpy()
    naive = float(joined["rv_fwd_ann"].mean())  # type: ignore[arg-type]
    err_naive = np.log(naive / joined["rv_fwd_ann"].to_numpy())
    assert float(np.mean(err_har**2)) < 0.8 * float(np.mean(err_naive**2))


def test_fit_refuses_underdetermined_samples(config: Config) -> None:
    frame = pl.DataFrame(
        {"log_rv_1d": [1.0, 2.0], "log_rv_5d": [1.0, 2.0], "rv_fwd_ann": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="complete rows"):
        fit_har(frame, ("log_rv_1d", "log_rv_5d"), "rv_fwd_ann")
