"""End-to-end smoke tests for both execution modes."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csp_forecaster import ConformalSeasonalPool   # noqa: E402


def _series(seed=0, n=400, m=24):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / m) + 0.05 * t + rng.normal(0, 0.5, n)


@pytest.mark.parametrize("mode", ["legacy", "fast"])
@pytest.mark.parametrize("adaptive", [True, False])
def test_predict_produces_valid_output(mode, adaptive):
    csp = ConformalSeasonalPool(adaptive=adaptive, mode=mode, random_state=0).fit(_series(), 24)
    r = csp.predict(24, alpha=0.05, n_samples=100)
    assert r.samples.shape == (24, 100)
    assert r.lower.shape == (24,) and r.upper.shape == (24,)
    assert np.all(np.isfinite(r.samples))
    assert np.all(r.lower <= r.upper)
    assert r.method == ("CSP-Adaptive" if adaptive else "CSP-Fixed")


def test_fast_samples_are_float32():
    r = ConformalSeasonalPool(mode="fast", random_state=0).fit(_series(), 24).predict(12, n_samples=50)
    assert r.samples.dtype == np.float32


def test_handles_non_seasonal_series():
    # m=1: no seasonal pool; both modes should still return a full sample budget.
    for mode in ("legacy", "fast"):
        r = ConformalSeasonalPool(mode=mode, random_state=0).fit(_series(m=1), seasonal_period=1).predict(10, n_samples=100)
        assert r.samples.shape == (10, 100)
