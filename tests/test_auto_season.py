"""seasonal_period="auto": the period is chosen on the history, and nothing else changes."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csp_forecaster import ConformalSeasonalPool, CSPModel, seasonal_candidates   # noqa: E402


def _randomwalk(seed=0, n=3000):
    return np.random.default_rng(seed).normal(0, 1, n).cumsum() + 100


def _seasonal(seed=0, n=3000, m=24):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / m) + rng.normal(0, 0.2, n)


def test_random_walk_with_declared_long_season_picks_no_season():
    # "minute data, declared daily season": the last value is the best anchor.
    y = _randomwalk()
    csp = ConformalSeasonalPool(random_state=1).fit(y, "auto", candidates=[1, 60, 288])
    assert csp.seasonal_period == 1
    assert set(csp.season_scores_) == {1, 60, 288}
    assert csp.season_scores_[1] < csp.season_scores_[288]


def test_seasonal_series_picks_its_season():
    y = _seasonal()
    csp = ConformalSeasonalPool(random_state=1).fit(y, "auto", candidates=[1, 24, 168])
    assert csp.seasonal_period == 24


def test_auto_is_better_than_the_wrong_declared_season():
    y = _randomwalk(3, 4000)
    train, test = y[:3000], y[3000:]

    def width_and_cover(m, **kw):
        w, c = [], []
        for i in range(0, len(test), 10):
            hist = np.concatenate([train, test[:i]])
            r = ConformalSeasonalPool(random_state=1).fit(hist, m, **kw).predict(1, alpha=0.1, n_samples=200)
            w.append(r.upper[0] - r.lower[0])
            c.append(r.lower[0] <= test[i] <= r.upper[0])
        return np.mean(w), np.mean(c)

    w_decl, _ = width_and_cover(288)
    w_auto, c_auto = width_and_cover("auto", candidates=[1, 288])
    assert w_auto < 0.5 * w_decl
    assert c_auto > 0.8


def test_choice_uses_only_the_history_and_is_deterministic():
    y = _seasonal(5)
    a = ConformalSeasonalPool(random_state=1).fit(y, "auto", candidates=[1, 24])
    b = ConformalSeasonalPool(random_state=99).fit(y, "auto", candidates=[24, 1])
    assert a.season_scores_ == b.season_scores_
    assert a.seasonal_period == b.seasonal_period


def test_selection_does_not_consume_the_model_random_stream():
    # auto that picks m must forecast exactly like fit(y, m) with the same seed
    y = _seasonal(7)
    a = ConformalSeasonalPool(random_state=1).fit(y, "auto", candidates=[1, 24])
    b = ConformalSeasonalPool(random_state=1).fit(y, a.seasonal_period)
    ra, rb = a.predict(12, n_samples=100), b.predict(12, n_samples=100)
    assert np.array_equal(ra.samples, rb.samples)


def test_integer_period_is_unchanged_and_ignores_candidates():
    y = _seasonal(2)
    a = ConformalSeasonalPool(random_state=1).fit(y, 24).predict(6)
    b = ConformalSeasonalPool(random_state=1).fit(y, 24, candidates=[1, 7]).predict(6)
    assert np.array_equal(a.samples, b.samples)


def test_horizon_is_respected():
    y = _seasonal(4)
    csp = ConformalSeasonalPool(random_state=1).fit(y, "auto", candidates=[1, 24], horizon=24)
    assert csp.seasonal_period == 24


def test_short_history_falls_back():
    y = _seasonal(1, n=20, m=4)
    csp = ConformalSeasonalPool().fit(y, "auto", candidates=[1, 4, 12])
    assert csp.seasonal_period == 4          # longest with three full cycles
    assert csp.season_scores_ == {}
    assert ConformalSeasonalPool().fit(y[:2], "auto", candidates=[4]).seasonal_period == 1


def test_candidates_longer_than_half_the_data_are_skipped():
    y = _randomwalk(1, 400)
    csp = ConformalSeasonalPool().fit(y, "auto", candidates=[1, 300])
    assert set(csp.season_scores_) == {1}
    assert csp.seasonal_period == 1


def test_errors():
    y = _seasonal()
    with pytest.raises(ValueError, match="freq"):
        ConformalSeasonalPool().fit(y, "auto")
    with pytest.raises(ValueError, match="unknown frequency"):
        ConformalSeasonalPool().fit(y, "auto", freq="fortnight")
    with pytest.raises(ValueError, match="auto"):
        ConformalSeasonalPool().fit(y, "weekly", candidates=[1])
    with pytest.raises(ValueError, match="season_candidates"):
        CSPModel(season_length="auto")


@pytest.mark.parametrize("freq,expected", [
    ("min", [1, 60, 1440, 10080]), ("T", [1, 60, 1440, 10080]), ("5min", [1, 12, 288, 2016]),
    ("15T", [1, 4, 96, 672]), ("30min", [1, 2, 48, 336]), ("10S", [1, 6, 360, 8640]),
    ("H", [1, 24, 168]), ("h", [1, 24, 168]), ("2H", [1, 12, 84]),
    ("D", [1, 7]), ("B", [1, 5]), ("W", [1, 52]), ("W-SUN", [1, 52]),
    ("M", [1, 12]), ("MS", [1, 12]), ("ME", [1, 12]), ("Q", [1, 4]), ("Q-DEC", [1, 4]),
    ("Y", [1]), ("A", [1]), ("YS", [1]),
])
def test_default_candidates_by_frequency(freq, expected):
    assert seasonal_candidates(freq) == expected


def test_auto_from_frequency_alone():
    # hourly data with a daily cycle: the default candidates are none, day, week
    y = _seasonal()
    csp = ConformalSeasonalPool(random_state=1).fit(y, "auto", freq="H")
    assert csp.seasonal_period == 24
    assert set(csp.season_scores_) == {1, 24, 168}


def test_own_candidates_are_added_to_the_frequency_defaults():
    y = _seasonal(m=30)
    csp = ConformalSeasonalPool(random_state=1).fit(y, "auto", freq="H", candidates=[30])
    assert set(csp.season_scores_) == {1, 24, 30, 168}
    assert csp.seasonal_period == 30


def test_nixtla_adapter_auto_from_freq():
    m = CSPModel(season_length="auto", freq="H", random_state=1)
    assert m.new().freq == "H"
    assert m.fit(_seasonal()).model_.seasonal_period == 24


def test_nixtla_adapter_auto():
    y = _randomwalk(2)
    m = CSPModel(season_length="auto", season_candidates=[1, 288], random_state=1)
    out = m.forecast(y, h=3, level=[90], fitted=True)
    assert out["mean"].shape == (3,)
    assert np.isnan(out["fitted"][0]) and not np.isnan(out["fitted"][1])   # chosen season 1
    fitted = m.new().fit(y)
    assert fitted.model_.seasonal_period == 1
    assert fitted.new().season_candidates == [1, 288]
    assert fitted.predict_in_sample()["fitted"].shape == y.shape
