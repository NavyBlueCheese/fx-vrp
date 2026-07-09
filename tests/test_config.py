from __future__ import annotations

from datetime import date
from pathlib import Path

from fxvrp.config import Config


def test_default_config_loads_fully_typed(config: Config) -> None:
    assert isinstance(config.root, Path)
    assert config.dukascopy.instrument == "EURUSD"
    assert config.dukascopy.price_scale > 0.0
    assert config.dukascopy.start_date < config.dukascopy.end_date
    assert isinstance(config.dukascopy.start_date, date)
    assert config.fred.series  # non-empty
    assert "EVZCLS" in config.fred.series
    assert "FXE" in config.cboe.symbols
    assert "_SPX" in config.cboe.symbols
    assert config.chain_cleaning.min_days_to_expiry >= 1
    assert config.quality.min_ticks_per_day > 0


def test_paths_are_anchored_at_repo_root(config: Config) -> None:
    for path in (
        config.paths.raw_dir,
        config.paths.interim_dir,
        config.paths.processed_dir,
        config.paths.reports_dir,
    ):
        assert path.is_absolute()
        assert config.root in path.parents
