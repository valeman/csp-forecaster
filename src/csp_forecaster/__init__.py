"""csp_forecaster -- training-free Conformal Seasonal Pools forecaster.

Packaged from the paper code (src/cp_bench/methods.py::ConformalSeasonalPool).
"""

from .core import (
    ConformalSeasonalPool,
    PredictionResult,
    DEFAULT_QUANTILE_LEVELS,
)
from .nixtla import CSPModel

__all__ = [
    "ConformalSeasonalPool",
    "PredictionResult",
    "DEFAULT_QUANTILE_LEVELS",
    "CSPModel",
]
__version__ = "0.1.0"
