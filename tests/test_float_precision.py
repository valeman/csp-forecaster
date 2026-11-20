"""Regression tests for sample precision (fixed in 0.2.1).

``mode="fast"`` used to return float32 samples. On flat or exactly periodic series
the degenerate interval then sat on the float32-rounded value and excluded the true
float64 value (coverage 0.0); on large-level, low-noise series the predictive spread
was rounded away entirely.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csp_forecaster import ConformalSeasonalPool   # noqa: E402

MODES = ["legacy", "fast"]


def _coverage(y, m, H, mode):
    hist, fut = y[:-H], y[-H:]
    r = ConformalSeasonalPool(mode=mode, random_state=0).fit(hist, seasonal_period=m)
    r = r.predict(H, alpha=0.05, n_samples=200)
    return float(np.mean((fut >= r.lower) & (fut <= r.upper))), r


@pytest.mark.parametrize("mode", MODES)
def test_samples_and_bounds_are_float64(mode):
    _, r = _coverage(np.linspace(1.0, 2.0, 80), 1, 6, mode)
    assert r.samples.dtype == np.float64
    assert np.asarray(r.lower).dtype == np.float64
    assert np.asarray(r.upper).dtype == np.float64


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("value", [0.1, 1234567.891, 16777217.0])
def test_flat_series_is_covered(mode, value):
    # None of these values is exactly representable in float32.
    assert float(np.float32(value)) != value
    cov, r = _coverage(np.full(60, value), 1, 6, mode)
    assert cov == 1.0
    assert np.all(r.samples == value)


@pytest.mark.parametrize("mode", MODES)
def test_exactly_periodic_series_is_covered(mode):
    cov, _ = _coverage(np.tile([0.1, 0.2, 0.3, 0.7], 20), 4, 8, mode)
    assert cov == 1.0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("level,sd", [(1e8, 0.5), (1e9, 5.0)])
def test_large_level_low_noise_keeps_its_spread(mode, level, sd):
    # float32 spacing is 8 at 1e8 and 64 at 1e9 -- wider than the noise itself.
    rng = np.random.default_rng(1)
    _, r = _coverage(level + rng.normal(0.0, sd, 300), 1, 6, mode)
    width = np.asarray(r.upper) - np.asarray(r.lower)
    assert np.all(width > sd)            # an interval, not a point
    assert np.all(width < 20 * sd)       # and on the scale of the noise
