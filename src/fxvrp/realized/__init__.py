"""Realised-variance estimators, validated against `fxvrp.simulate` ground truth.

Nothing in this package is trusted until it recovers a known answer in a synthetic
world: RV must converge to σ²T under GBM and to the path's ∫v dt under Heston, the
noise-robust estimators must survive contamination that breaks naive RV, and
bipower variation must strip exactly the jumps that Merton paths carry.
"""

from fxvrp.realized.estimators import (
    realized_kernel,
    realized_semivariance,
    realized_variance,
    subsampled_rv,
    tsrv,
)
from fxvrp.realized.jumps import (
    bipower_variation,
    bns_test_statistic,
    jump_variation,
    tripower_quarticity,
)

__all__ = [
    "bipower_variation",
    "bns_test_statistic",
    "jump_variation",
    "realized_kernel",
    "realized_semivariance",
    "realized_variance",
    "subsampled_rv",
    "tripower_quarticity",
    "tsrv",
]
