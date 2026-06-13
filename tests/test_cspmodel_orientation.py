"""Regression tests for the v0.1.2 ``CSPModel`` orientation fix.

Before v0.1.2, ``CSPModel._result_dict`` called plain ``np.quantile``
on the raw sample array, bypassing the orientation-correct conformal
quantile introduced in v0.1.1's ``_finalize``. As a result, users
calling CSP through the ``statsforecast``-compatible wrapper received
the same anti-conservative lower bound v0.1.1 was designed to remove.

These tests assert that ``CSPModel``'s ``lo-L`` / ``hi-L`` outputs are
byte-equivalent to ``ConformalSeasonalPool.predict(...).lower /
.upper`` so a future regression in the wrapper is caught immediately.
"""

import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # allow tests to be discovered without pytest
    class _Pytest:
        @staticmethod
        def approx(v, abs=1e-12):
            class _A:
                def __init__(_self, v_, a_): _self.v=v_; _self.a=a_
                def __eq__(_self, o): return abs(o - _self.v) <= _self.a
            return _A(v, abs)
    pytest = _Pytest()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csp_forecaster import ConformalSeasonalPool, CSPModel   # noqa: E402


def _series(seed=0, n=1500, m=24):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / m) + rng.normal(0, 1, n)


def test_cspmodel_lo_95_matches_core_lower():
    """At alpha=0.05, the adapter's lo-95 / hi-95 must be byte-identical
    to ``ConformalSeasonalPool.predict(alpha=0.05).lower / .upper``.
    """
    y = _series()
    core = ConformalSeasonalPool(
        adaptive=True, mode="fast", random_state=42,
    ).fit(y, 24)
    r_core = core.predict(H=100, alpha=0.05, n_samples=100)

    adapter = CSPModel(
        season_length=24, mode="fast", random_state=42,
    ).fit(y)
    r_adapter = adapter.predict(h=100, level=[95])

    np.testing.assert_allclose(r_adapter["lo-95"], r_core.lower, atol=1e-12)
    np.testing.assert_allclose(r_adapter["hi-95"], r_core.upper, atol=1e-12)


def test_cspmodel_lo_90_matches_core_lower():
    """Same property at alpha=0.10 / level=90."""
    y = _series(seed=1)
    core = ConformalSeasonalPool(
        adaptive=True, mode="fast", random_state=42,
    ).fit(y, 24)
    r_core = core.predict(H=50, alpha=0.10, n_samples=100)

    adapter = CSPModel(
        season_length=24, mode="fast", random_state=42,
    ).fit(y)
    r_adapter = adapter.predict(h=50, level=[90])

    np.testing.assert_allclose(r_adapter["lo-90"], r_core.lower, atol=1e-12)
    np.testing.assert_allclose(r_adapter["hi-90"], r_core.upper, atol=1e-12)


def test_cspmodel_multi_level_monotone():
    """Cross-level monotonicity: wider intervals contain narrower ones.
    A direct consequence of the orientation fix being applied per-level.
    """
    y = _series(seed=2)
    adapter = CSPModel(
        season_length=24, mode="fast", random_state=42,
    ).fit(y)
    r = adapter.predict(h=40, level=[80, 90, 95])
    assert np.all(r["lo-80"] >= r["lo-90"] - 1e-9)
    assert np.all(r["lo-90"] >= r["lo-95"] - 1e-9)
    assert np.all(r["hi-80"] <= r["hi-90"] + 1e-9)
    assert np.all(r["hi-90"] <= r["hi-95"] + 1e-9)


def test_cspmodel_forecast_matches_predict():
    """forecast(y, h, level) is fit-then-predict — should match
    fit().predict() to floating-point precision."""
    y = _series(seed=3)
    a1 = CSPModel(season_length=24, mode="fast", random_state=42)
    a2 = CSPModel(season_length=24, mode="fast", random_state=42)
    a1.fit(y)
    r_via_fit = a1.predict(h=30, level=[95])
    r_via_forecast = a2.forecast(y, h=30, level=[95])
    np.testing.assert_allclose(r_via_fit["lo-95"], r_via_forecast["lo-95"], atol=1e-12)
    np.testing.assert_allclose(r_via_fit["hi-95"], r_via_forecast["hi-95"], atol=1e-12)
