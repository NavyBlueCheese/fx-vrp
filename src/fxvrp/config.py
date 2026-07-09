"""Typed access to ``configs/default.yaml``.

All numeric assumptions live in the YAML file; this module gives them a frozen,
fully-typed shape so that a typo in a config key fails loudly at load time rather
than propagating a default silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class DukascopyConfig:
    base_url: str
    instrument: str
    price_scale: float
    pip: float
    start_date: date
    end_date: date
    sample_start_date: date
    sample_end_date: date
    max_retries: int
    retry_backoff_s: float
    request_timeout_s: float
    throttle_s: float


@dataclass(frozen=True)
class FredConfig:
    base_url: str
    series: tuple[str, ...]


@dataclass(frozen=True)
class EcbConfig:
    base_url: str
    estr_series: str
    estr_start: date


@dataclass(frozen=True)
class CboeConfig:
    base_url: str
    symbols: tuple[str, ...]
    request_timeout_s: float
    throttle_s: float


@dataclass(frozen=True)
class ChainCleaningConfig:
    min_bid: float
    min_days_to_expiry: int
    max_days_to_expiry: int


@dataclass(frozen=True)
class QualityConfig:
    min_ticks_per_day: int
    stale_run_s: float
    max_plausible_spread_pips: float
    max_gap_report_s: float
    weekend_gap_min_hours: float


@dataclass(frozen=True)
class RealizedConfig:
    grid_interval_s: int
    signature_intervals_s: tuple[int, ...]
    tsrv_subgrids: int
    kernel_sparse_interval_s: int
    min_returns_per_day: int
    day_close_local: str
    day_close_tz: str


@dataclass(frozen=True)
class SimulateConfig:
    default_seed: int


@dataclass(frozen=True)
class Config:
    root: Path
    paths: PathsConfig
    dukascopy: DukascopyConfig
    fred: FredConfig
    ecb: EcbConfig
    cboe: CboeConfig
    chain_cleaning: ChainCleaningConfig
    quality: QualityConfig
    realized: RealizedConfig
    simulate: SimulateConfig


def _str(section: dict[str, Any], key: str) -> str:
    value = section[key]
    if not isinstance(value, str):
        raise TypeError(f"config key {key!r} must be a string, got {type(value).__name__}")
    return value


def _float(section: dict[str, Any], key: str) -> float:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"config key {key!r} must be a number, got {type(value).__name__}")
    return float(value)


def _int(section: dict[str, Any], key: str) -> int:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"config key {key!r} must be an integer, got {type(value).__name__}")
    return value


def _date(section: dict[str, Any], key: str) -> date:
    value = section[key]
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"config key {key!r} must be an ISO date, got {type(value).__name__}")


def _str_tuple(section: dict[str, Any], key: str) -> tuple[str, ...]:
    value = section[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"config key {key!r} must be a list of strings")
    return tuple(value)


def _int_tuple(section: dict[str, Any], key: str) -> tuple[int, ...]:
    value = section[key]
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"config key {key!r} must be a list of integers")
    return tuple(value)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw[key]
    if not isinstance(value, dict):
        raise TypeError(f"config section {key!r} must be a mapping")
    return value


def load_config(path: Path | None = None) -> Config:
    """Load and validate the project configuration."""
    config_path = path if path is not None else DEFAULT_CONFIG_PATH
    root = config_path.resolve().parents[1]
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("top-level config must be a mapping")

    paths = _section(raw, "paths")
    duka = _section(raw, "dukascopy")
    fred = _section(raw, "fred")
    ecb = _section(raw, "ecb")
    cboe = _section(raw, "cboe")
    cleaning = _section(raw, "chain_cleaning")
    quality = _section(raw, "quality")
    realized = _section(raw, "realized")
    simulate = _section(raw, "simulate")

    return Config(
        root=root,
        paths=PathsConfig(
            raw_dir=root / _str(paths, "raw_dir"),
            interim_dir=root / _str(paths, "interim_dir"),
            processed_dir=root / _str(paths, "processed_dir"),
            reports_dir=root / _str(paths, "reports_dir"),
        ),
        dukascopy=DukascopyConfig(
            base_url=_str(duka, "base_url"),
            instrument=_str(duka, "instrument"),
            price_scale=_float(duka, "price_scale"),
            pip=_float(duka, "pip"),
            start_date=_date(duka, "start_date"),
            end_date=_date(duka, "end_date"),
            sample_start_date=_date(duka, "sample_start_date"),
            sample_end_date=_date(duka, "sample_end_date"),
            max_retries=_int(duka, "max_retries"),
            retry_backoff_s=_float(duka, "retry_backoff_s"),
            request_timeout_s=_float(duka, "request_timeout_s"),
            throttle_s=_float(duka, "throttle_s"),
        ),
        fred=FredConfig(
            base_url=_str(fred, "base_url"),
            series=_str_tuple(fred, "series"),
        ),
        ecb=EcbConfig(
            base_url=_str(ecb, "base_url"),
            estr_series=_str(ecb, "estr_series"),
            estr_start=_date(ecb, "estr_start"),
        ),
        cboe=CboeConfig(
            base_url=_str(cboe, "base_url"),
            symbols=_str_tuple(cboe, "symbols"),
            request_timeout_s=_float(cboe, "request_timeout_s"),
            throttle_s=_float(cboe, "throttle_s"),
        ),
        chain_cleaning=ChainCleaningConfig(
            min_bid=_float(cleaning, "min_bid"),
            min_days_to_expiry=_int(cleaning, "min_days_to_expiry"),
            max_days_to_expiry=_int(cleaning, "max_days_to_expiry"),
        ),
        quality=QualityConfig(
            min_ticks_per_day=_int(quality, "min_ticks_per_day"),
            stale_run_s=_float(quality, "stale_run_s"),
            max_plausible_spread_pips=_float(quality, "max_plausible_spread_pips"),
            max_gap_report_s=_float(quality, "max_gap_report_s"),
            weekend_gap_min_hours=_float(quality, "weekend_gap_min_hours"),
        ),
        realized=RealizedConfig(
            grid_interval_s=_int(realized, "grid_interval_s"),
            signature_intervals_s=_int_tuple(realized, "signature_intervals_s"),
            tsrv_subgrids=_int(realized, "tsrv_subgrids"),
            kernel_sparse_interval_s=_int(realized, "kernel_sparse_interval_s"),
            min_returns_per_day=_int(realized, "min_returns_per_day"),
            day_close_local=_str(realized, "day_close_local"),
            day_close_tz=_str(realized, "day_close_tz"),
        ),
        simulate=SimulateConfig(default_seed=_int(simulate, "default_seed")),
    )
