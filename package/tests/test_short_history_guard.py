"""short_history_guard: no seasonal period when the history is shorter than one period."""
import numpy as np

from csp_forecaster import ConformalSeasonalPool


def _walk(n, seed=0):
    return 50 + np.cumsum(np.random.default_rng(seed).standard_normal(n))


def test_fires_below_one_season_and_matches_no_season():
    y = _walk(700)
    g = ConformalSeasonalPool(random_state=3).fit(y, 1440)
    assert g.short_history_guard_active_ and g.seasonal_period == 1 and g.seasonal_period_given_ == 1440
    a = g.predict(H=48, alpha=0.05, n_samples=200)
    b = ConformalSeasonalPool(random_state=3).fit(y, 1).predict(H=48, alpha=0.05, n_samples=200)
    np.testing.assert_array_equal(a.lower, b.lower)
    np.testing.assert_array_equal(a.upper, b.upper)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_does_not_fire_with_one_full_season():
    y = _walk(1440)
    g = ConformalSeasonalPool(random_state=3).fit(y, 1440)
    assert not g.short_history_guard_active_ and g.seasonal_period == 1440
    y = _walk(600)
    g = ConformalSeasonalPool(random_state=3).fit(y, 24)
    assert not g.short_history_guard_active_ and g.seasonal_period == 24


def test_can_be_switched_off():
    y = _walk(700)
    g = ConformalSeasonalPool(random_state=3, short_history_guard=False).fit(y, 1440)
    assert not g.short_history_guard_active_ and g.seasonal_period == 1440


def test_legacy_mode_is_untouched():
    y = _walk(20)
    g = ConformalSeasonalPool(mode="legacy").fit(y, 24)
    assert not g.short_history_guard_active_ and g.seasonal_period == 24


def test_widths_grow_with_step_when_guard_fires():
    """Without the guard the width is flat across steps on a random walk; with it, it grows."""
    y = _walk(700, seed=5)
    on = ConformalSeasonalPool(random_state=1).fit(y, 1440).predict(H=48, alpha=0.05, n_samples=400)
    w = np.asarray(on.upper) - np.asarray(on.lower)
    assert w[-12:].mean() > 1.5 * w[:6].mean()


def test_guard_improves_coverage_on_short_persistent_series():
    hits_on = hits_off = total = 0
    for seed in range(40):
        y = _walk(760, seed=seed)
        h, f = y[:712], y[712:]
        for flag in (True, False):
            r = ConformalSeasonalPool(random_state=seed, short_history_guard=flag).fit(h, 1440).predict(H=48, alpha=0.05, n_samples=200)
            c = int(np.sum((f >= r.lower) & (f <= r.upper)))
            if flag:
                hits_on += c
            else:
                hits_off += c
        total += 48
    assert hits_on / total > 0.85
    assert hits_on > hits_off


def test_refit_resets_the_flag():
    g = ConformalSeasonalPool(random_state=3).fit(_walk(100), 365)
    assert g.short_history_guard_active_
    g.fit(_walk(800), 24)
    assert not g.short_history_guard_active_ and g.seasonal_period == 24


def test_with_backtest_selection():
    y = _walk(700)
    g = ConformalSeasonalPool(random_state=3, season_selection="backtest").fit(y, 1440)
    g.predict(H=48)
    assert g.seasonal_period_used_ == 1
