"""CSPAuto (0.3.0): the target (levels or one-step changes) and the seasonal period chosen on the history."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csp_forecaster import ConformalSeasonalPool, CSPAuto, acf_peak, auto_candidates   # noqa: E402


def _randomwalk(seed=0, n=3000):
    return np.random.default_rng(seed).normal(0, 1, n).cumsum() + 100


def _seasonal(seed=0, n=3000, m=24):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / m) + rng.normal(0, 0.2, n)


def test_candidates():
    assert auto_candidates(1) == [1]
    assert auto_candidates(7) == [1, 7]
    assert auto_candidates(288) == [1, 12, 288]
    assert auto_candidates(1440) == [1, 60, 1440]
    assert auto_candidates(48) == [1, 48]          # below 96: no hour candidate


def test_random_walk_with_a_declared_daily_season_drops_the_season():
    # plain CSP with no season already anchors on the last value: that is the right model here
    m = CSPAuto(random_state=1).fit(_randomwalk(), 288)
    assert m.seasonal_period_ == 1
    assert m.choice_["scores"]["level_1"] < m.choice_["scores"]["level_288"]


def test_seasonal_changes_choose_the_change_target():
    # the series wanders, but its changes repeat each day: last value plus the seasonal change
    rng = np.random.default_rng(2)
    t = np.arange(4000)
    y = 50 + np.cumsum(np.sin(2 * np.pi * t / 24) + 0.3 * rng.normal(size=4000))
    m = CSPAuto(random_state=1).fit(y, 24)
    assert (m.target_, m.seasonal_period_) == ("change", 24)
    assert m.choice_["scores"]["change_24"] < m.choice_["scores"]["level_24"]


def test_stable_seasonal_series_keeps_its_season():
    m = CSPAuto(random_state=1).fit(_seasonal(), 24)
    assert m.seasonal_period_ == 24


def test_change_forecast_is_anchored_on_the_last_value():
    y = _randomwalk(4)
    m = CSPAuto(random_state=1, targets=("change",)).fit(y, 1)
    r = m.predict(1, alpha=0.1, n_samples=400)
    inner = ConformalSeasonalPool(random_state=1).fit(np.diff(y), 1).predict(1, alpha=0.1, n_samples=400)
    assert np.allclose(r.samples, y[-1] + inner.samples)
    assert np.allclose(r.lower, y[-1] + inner.lower) and np.allclose(r.upper, y[-1] + inner.upper)


def test_multi_step_change_paths_are_cumulative():
    y = _randomwalk(5)
    m = CSPAuto(random_state=2, targets=("change",)).fit(y, 1)
    r = m.predict(5, alpha=0.1, n_samples=500)
    w = r.upper - r.lower
    assert np.all(np.diff(w) > 0)                  # a wandering series: wider further out
    assert r.samples.shape == (5, 500)


def test_levels_only_equals_plain_csp_with_the_chosen_period():
    y = _seasonal(3)
    a = CSPAuto(random_state=1, targets=("level",)).fit(y, 24)
    b = ConformalSeasonalPool(random_state=1).fit(y, a.seasonal_period_)
    ra, rb = a.predict(6, n_samples=100), b.predict(6, n_samples=100)
    assert np.array_equal(ra.samples, rb.samples)


def test_short_history_falls_back_to_plain_csp():
    y = _seasonal(1, n=60, m=12)
    m = CSPAuto(random_state=1).fit(y, 12)
    assert (m.target_, m.seasonal_period_) == ("level", 12) and m.choice_["scores"] == {}


def test_choice_is_deterministic_and_seed_free():
    y = _randomwalk(7)
    a = CSPAuto(random_state=1).fit(y, 24)
    b = CSPAuto(random_state=99).fit(y, 24)
    assert a.choice_ == b.choice_


def test_acf_peak_finds_a_short_cycle():
    rng = np.random.default_rng(0)
    t = np.arange(4000)
    y = np.sin(2 * np.pi * t / 6) + 0.2 * rng.normal(size=4000)
    assert acf_peak(y) == 6
    assert acf_peak(rng.normal(size=4000)) is None
    m = CSPAuto(acf_peak=True, random_state=1).fit(y, 1)    # declared: no season
    assert m.choice_["acf_peak"] == 6 and m.seasonal_period_ == 6


def test_bad_arguments():
    with pytest.raises(ValueError):
        CSPAuto(targets=("levels",))
    with pytest.raises(RuntimeError):
        CSPAuto().predict(1)
