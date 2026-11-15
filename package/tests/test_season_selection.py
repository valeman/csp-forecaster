"""season_selection="backtest": choose between the given seasonal period and none."""
import numpy as np
import pytest

from csp_forecaster import ConformalSeasonalPool


def _seasonal(n=600, m=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 5 * np.sin(2 * np.pi * t / m) + 0.3 * rng.standard_normal(n)


def _walk(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return 50 + np.cumsum(rng.standard_normal(n))


def test_default_is_unchanged():
    y = _seasonal()
    a = ConformalSeasonalPool(random_state=1).fit(y, 24).predict(H=12, alpha=0.05, n_samples=200)
    b = ConformalSeasonalPool(random_state=1, season_selection="given").fit(y, 24).predict(H=12, alpha=0.05, n_samples=200)
    np.testing.assert_array_equal(a.lower, b.lower)
    np.testing.assert_array_equal(a.upper, b.upper)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_keeps_a_real_season():
    y = _seasonal()
    f = ConformalSeasonalPool(random_state=1, season_selection="backtest").fit(y, 24)
    r = f.predict(H=12, alpha=0.05, n_samples=200)
    assert f.seasonal_period_used_ == 24 and f.seasonal_period_given_ == 24
    g = ConformalSeasonalPool(random_state=1).fit(y, 24).predict(H=12, alpha=0.05, n_samples=200)
    np.testing.assert_array_equal(r.lower, g.lower)        # same seed, same season: identical forecast


def test_drops_a_season_the_series_does_not_have():
    y = _walk()
    f = ConformalSeasonalPool(random_state=1, season_selection="backtest").fit(y, 200)
    r = f.predict(H=12, alpha=0.05, n_samples=200)
    assert f.seasonal_period_used_ == 1
    g = ConformalSeasonalPool(random_state=1).fit(y, 1).predict(H=12, alpha=0.05, n_samples=200)
    np.testing.assert_array_equal(r.lower, g.lower)
    np.testing.assert_array_equal(r.upper, g.upper)


def test_too_short_to_backtest_keeps_given():
    y = _seasonal(n=40, m=12)
    f = ConformalSeasonalPool(random_state=1, season_selection="backtest").fit(y, 12)
    f.predict(H=12, alpha=0.05)
    assert f.seasonal_period_used_ == 12


def test_no_season_given_is_a_no_op_and_refit_resets():
    y = _walk()
    f = ConformalSeasonalPool(random_state=1, season_selection="backtest").fit(y, 1)
    f.predict(H=6)
    assert f.seasonal_period_used_ == 1
    f.fit(_seasonal(), 24).predict(H=12)
    assert f.seasonal_period_given_ == 24 and f.seasonal_period_used_ == 24


def test_rejects_unknown_option():
    with pytest.raises(ValueError):
        ConformalSeasonalPool(season_selection="auto")
