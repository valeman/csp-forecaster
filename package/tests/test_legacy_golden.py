"""Legacy-path determinism (golden regression).

The legacy mode draws from NumPy's global RNG, so under a fixed global seed it is
deterministic: two runs with the same seed and history produce identical samples
and identical quantiles.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csp_forecaster import ConformalSeasonalPool   # noqa: E402


def _series(seed=0, n=400, m=24):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / m) + 0.05 * t + rng.normal(0, 0.5, n)


def test_legacy_deterministic_under_fixed_global_seed():
    hist = _series()
    a = ConformalSeasonalPool(mode="legacy").fit(hist, 24)
    b = ConformalSeasonalPool(mode="legacy").fit(hist, 24)
    np.random.seed(123)
    ra = a.predict(24, n_samples=100)
    np.random.seed(123)
    rb = b.predict(24, n_samples=100)
    np.testing.assert_array_equal(ra.samples, rb.samples)
    for tau in ra.quantiles:
        np.testing.assert_array_equal(ra.quantiles[tau], rb.quantiles[tau])
