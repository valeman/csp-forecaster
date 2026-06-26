"""Tests for the residual_mode and orientation options."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csp_forecaster import ConformalSeasonalPool   # noqa: E402


def _seasonal(seed=0, n=600, m=24):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / m) + 0.02 * t + rng.normal(0, 0.4, n)


def _randomwalk(seed=0, n=600):
    return np.random.default_rng(seed).normal(0, 1, n).cumsum() + 100


# ----------------------------------------------------------------- defaults
def test_defaults_are_recommended_config():
    # v0.1.4: defaults are the best-scoring config, not the paper baseline.
    m = ConformalSeasonalPool()
    assert m.residual_mode == "h_step"
    assert m.orientation is False
    assert m.decay_unit == "step"


def test_invalid_residual_mode_raises():
    with pytest.raises(ValueError, match="residual_mode"):
        ConformalSeasonalPool(residual_mode="bogus")


def test_invalid_decay_unit_raises():
    with pytest.raises(ValueError, match="decay_unit"):
        ConformalSeasonalPool(decay_unit="bogus")


def test_decay_unit_changes_seasonal_pool():
    # step vs cycle weight the same-phase pool differently -> different samples.
    y = _seasonal(m=24)
    a = ConformalSeasonalPool(mode="fast", random_state=0, decay_unit="step").fit(y, 24).predict(24, n_samples=4000)
    b = ConformalSeasonalPool(mode="fast", random_state=0, decay_unit="cycle").fit(y, 24).predict(24, n_samples=4000)
    assert not np.allclose(a.samples.mean(1), b.samples.mean(1), atol=1e-6)


# -------------------------------------------------------------- orientation
def test_orientation_only_affects_quantiles_not_samples():
    y = _seasonal()
    on = ConformalSeasonalPool(mode="fast", random_state=0, orientation=True).fit(y, 24).predict(24, n_samples=100)
    off = ConformalSeasonalPool(mode="fast", random_state=0, orientation=False).fit(y, 24).predict(24, n_samples=100)
    np.testing.assert_array_equal(on.samples, off.samples)         # same draws
    assert np.mean(on.upper - on.lower) > np.mean(off.upper - off.lower)  # on is wider


# ---------------------------------------------------------------- residual_mode
def test_hstep_is_noop_when_horizon_within_season():
    y = _seasonal(m=24)
    p = ConformalSeasonalPool(mode="fast", random_state=3, residual_mode="paper", orientation=False).fit(y, 24).predict(24, n_samples=100).samples
    s = ConformalSeasonalPool(mode="fast", random_state=3, residual_mode="h_step", orientation=False).fit(y, 24).predict(24, n_samples=100).samples
    np.testing.assert_array_equal(p, s)


def test_hstep_widens_with_horizon_for_nonseasonal():
    hist = _randomwalk(0, 600)
    rp = ConformalSeasonalPool(adaptive=False, mode="fast", random_state=0, residual_mode="paper", orientation=False).fit(hist, 1).predict(30, n_samples=2000)
    rh = ConformalSeasonalPool(adaptive=False, mode="fast", random_state=0, residual_mode="h_step", orientation=False).fit(hist, 1).predict(30, n_samples=2000)
    wp, wh = rp.upper - rp.lower, rh.upper - rh.lower
    assert wh[-1] > 1.5 * wh[0]                  # h_step widens with horizon
    assert wp[-1] == pytest.approx(wp[0], rel=0.5)  # paper ~flat
    assert wh[-1] > wp[-1]
