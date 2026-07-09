"""Synthetic worlds with analytically known answers.

Built before, and independently of, every estimator they validate: an estimator is
trusted only after it recovers the truth in one of these worlds.
"""

from fxvrp.simulate.gbm import GBMPaths, simulate_gbm
from fxvrp.simulate.heston import HestonParams, HestonPaths, simulate_heston
from fxvrp.simulate.merton import MertonParams, MertonPaths, simulate_merton
from fxvrp.simulate.microstructure import add_bid_ask_bounce, add_gaussian_noise

__all__ = [
    "GBMPaths",
    "HestonParams",
    "HestonPaths",
    "MertonParams",
    "MertonPaths",
    "add_bid_ask_bounce",
    "add_gaussian_noise",
    "simulate_gbm",
    "simulate_heston",
    "simulate_merton",
]
