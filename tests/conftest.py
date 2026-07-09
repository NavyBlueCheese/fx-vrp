from __future__ import annotations

import numpy as np
import pytest

from fxvrp.config import Config, load_config


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config()


@pytest.fixture()
def rng(config: Config) -> np.random.Generator:
    return np.random.default_rng(config.simulate.default_seed)
