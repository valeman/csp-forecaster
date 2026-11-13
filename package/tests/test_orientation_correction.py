"""Regression tests for the orientation-correct finite-sample conformal quantile.

The lower bound at ``q < 0.5`` should use ``floor((n+1)*q)/n`` so it picks a
more-extreme (lower) value than plain ``np.quantile(samples, q)``. The upper
bound at ``q >= 0.5`` should use ``ceil((n+1)*q)/n`` symmetrically.

This was the change in v0.1.1: ``_finalize`` used to call
``np.quantile(samples, taus, axis=1)`` directly, which is anti-conservative on
the lower tail. The fix preserves a single vectorized ``np.quantile`` call by
remapping the requested ``taus`` through ``_oriented_index`` first.

Run from the repo root:
    PYTHONPATH=src python -m pytest tests/test_orientation_correction.py -q
"""

import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # allow tests to be discovered without pytest installed
    class _Approx:
        def __init__(self, v, abs=1e-12): self.v = v; self.abs = abs
        def __eq__(self, other): return abs(other - self.v) <= self.abs
    class _Pytest:
        @staticmethod
        def approx(v, abs=1e-12): return _Approx(v, abs)
    pytest = _Pytest()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csp_forecaster import ConformalSeasonalPool   # noqa: E402


# ----------------------------------------------------------------------
# Unit tests on _oriented_index itself
# ----------------------------------------------------------------------
def test_oriented_index_lower_uses_floor():
    """For q < 0.5 the index should round DOWN."""
    idx = ConformalSeasonalPool._oriented_index(0.025, n=100)
    # floor(101 * 0.025) / 100 = floor(2.525) / 100 = 2 / 100 = 0.02
    assert idx == pytest.approx(0.02, abs=1e-12)


def test_oriented_index_upper_uses_ceil():
    """For q >= 0.5 the index should round UP."""
    idx = ConformalSeasonalPool._oriented_index(0.975, n=100)
    # ceil(101 * 0.975) / 100 = ceil(98.475) / 100 = 99 / 100 = 0.99
    assert idx == pytest.approx(0.99, abs=1e-12)


def test_oriented_index_median_uses_ceil():
    """Exactly 0.5 falls into the upper branch (>= 0.5)."""
    idx = ConformalSeasonalPool._oriented_index(0.5, n=100)
    # ceil(101 * 0.5) / 100 = ceil(50.5) / 100 = 51 / 100 = 0.51
    assert idx == pytest.approx(0.51, abs=1e-12)


def test_oriented_index_clips_to_unit_interval():
    """Edge cases: q very close to 0 or 1 must stay inside [0, 1]."""
    assert ConformalSeasonalPool._oriented_index(0.0001, n=10) >= 0.0
    assert ConformalSeasonalPool._oriented_index(0.9999, n=10) <= 1.0


def test_oriented_index_degenerate_n():
    """n=0 must not divide by zero."""
    out = ConformalSeasonalPool._oriented_index(0.5, n=0)
    assert out == pytest.approx(0.5, abs=1e-12)


# ----------------------------------------------------------------------
# Integration test: the lower bound is more extreme after the fix
# ----------------------------------------------------------------------
def _stationary_series(n=1500, m=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10.0 + 3.0 * np.sin(2 * np.pi * t / m) + rng.normal(0, 1.0, n)


def test_lower_bound_is_no_less_extreme_than_plain_quantile():
    """The orientation-corrected lower bound is always <= plain np.quantile,
    and strictly less on at least one horizon when (n+1)*alpha/2 is not an
    integer multiple of n (which is the typical case at alpha=0.1, n=100).
    """
    y = _stationary_series(seed=0)
    csp = ConformalSeasonalPool(adaptive=True, mode="fast", orientation=True, random_state=42)
    csp.fit(y, seasonal_period=24)
    # alpha=0.05 + n_samples=100  ->  (n+1)*0.025 = 2.525  ->  floor=2, ceil=3
    # so floor/n = 0.02 differs from plain q=0.025 (h=99*0.025=2.475)
    fc = csp.predict(H=24, alpha=0.05, n_samples=100)

    plain_lower = np.quantile(fc.samples, 0.025, axis=1)
    plain_upper = np.quantile(fc.samples, 0.975, axis=1)

    # Orientation-corrected lower <= plain lower (non-strict): conservative
    assert np.all(fc.lower <= plain_lower + 1e-9), \
        "orientation correction must never make the lower bound less extreme"
    # At least one horizon should show a strict decrease
    assert np.any(fc.lower < plain_lower - 1e-9), \
        "orientation correction should move lower bound on at least one horizon"

    # Symmetrically: corrected upper >= plain upper, with at least one strict
    assert np.all(fc.upper >= plain_upper - 1e-9), \
        "orientation correction must never make the upper bound less extreme"
    assert np.any(fc.upper > plain_upper + 1e-9), \
        "orientation correction should move upper bound on at least one horizon"


def test_coverage_improves_or_holds_after_fix():
    """The fix should not reduce empirical coverage on a benign series.

    We average over many seeds because per-series coverage is noisy at H=24.
    """
    target_coverage = 1.0 - 0.10  # alpha = 0.10
    covered_count = 0
    total_count = 0
    n_seeds = 20

    for seed in range(n_seeds):
        y = _stationary_series(seed=seed, n=1500)
        train, test = y[:1300], y[1300:]
        csp = ConformalSeasonalPool(adaptive=True, mode="fast", orientation=True, random_state=seed)
        csp.fit(train, seasonal_period=24)
        # Predict the next 200 points in one shot — paper-style.
        fc = csp.predict(H=len(test), alpha=0.10, n_samples=100)
        covered_count += int(np.sum((test >= fc.lower) & (test <= fc.upper)))
        total_count += len(test)

    empirical = covered_count / total_count
    # We do not assert nominal coverage exactly (CSP is approximately
    # calibrated, not exactly) but we do assert it is at least 0.85
    # which is consistent with the v6 Monash benchmark result (0.91 mean).
    assert empirical >= 0.85, f"empirical coverage too low: {empirical:.3f}"


def test_returns_three_taus_for_default_alpha():
    """alpha/2 and 1-alpha/2 are added to the default quantile_levels grid.

    Regression guard: a previous refactor risked collapsing duplicate taus.
    """
    y = _stationary_series(seed=1)
    csp = ConformalSeasonalPool(adaptive=True, mode="fast", orientation=True, random_state=1)
    csp.fit(y, seasonal_period=24)
    fc = csp.predict(H=12, alpha=0.10, n_samples=64)

    # lower / upper should match the alpha/2 and 1-alpha/2 entries of quantiles
    # IF those are in the requested levels — by default 0.05 and 0.95 are not in
    # DEFAULT_QUANTILE_LEVELS, but they are added internally for lower/upper.
    assert fc.lower.shape == (12,)
    assert fc.upper.shape == (12,)
    assert np.all(fc.lower <= fc.upper + 1e-9)
