"""Regression tests for the v0.2.0 residual-pool starvation fix.

Bug: with ``residual_mode="h_step"`` the lag grows as ``L_h = m * ceil(h / m)`` while the pool
was drawn from the calibration window only (half the history), so on short series the pool
shrank to one element and the predictive distribution collapsed to a point mass -- interval
width exactly 0.0, coverage exactly 0.0.
"""
import numpy as np
import pytest

from csp_forecaster import ConformalSeasonalPool


# A real short annual series: 14 observations, 6-step horizon, m=1.
# Pre-0.2.0 this produced pool sizes 6,5,4,3,2,1 and a zero-width interval at h=6.
SHORT_YEARLY = np.array(
    [3897.0, 4290.0, 4629.0, 5138.0, 5439.0, 5638.0, 5905.0, 6247.0,
     6415.0, 6650.0, 6839.0, 7143.0, 7452.0, 7891.0]
)


def test_no_zero_width_interval_on_short_series():
    """The headline bug: a 95% interval must never have zero width."""
    r = ConformalSeasonalPool(random_state=0).fit(SHORT_YEARLY, 1).predict(
        H=6, alpha=0.05, n_samples=200)
    width = np.asarray(r.upper) - np.asarray(r.lower)
    assert (width > 0).all(), f"zero-width interval at horizon(s) {np.where(width <= 0)[0] + 1}"


def test_interval_widens_with_horizon_on_short_series():
    """Multi-step uncertainty must not shrink as the horizon grows."""
    r = ConformalSeasonalPool(random_state=0).fit(SHORT_YEARLY, 1).predict(
        H=6, alpha=0.05, n_samples=400)
    width = np.asarray(r.upper) - np.asarray(r.lower)
    assert width[-1] > width[0], f"interval narrows with horizon: {width}"


def test_residual_pool_never_degenerate():
    """Every per-horizon pool must have at least MIN_RESIDUALS entries, or be scaled."""
    m = ConformalSeasonalPool().fit(SHORT_YEARLY, 1)
    for lag in range(1, 7):
        R = m._residuals_lag(lag)
        assert R.size >= 2, f"lag {lag} gave a degenerate pool of size {R.size}"
        assert np.isfinite(R).all()


@pytest.mark.parametrize("lag", [1, 2, 3, 6, 12])
def test_long_series_behaviour_unchanged(lag):
    """Branch 1 must fire on long histories, reproducing the pre-0.2.0 pool exactly."""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1, 500)) + 100.0
    mdl = ConformalSeasonalPool().fit(y, 12)

    T = len(y)
    n_cal = min(max(int(T * mdl.cal_fraction), mdl.seasonal_period + 1), T)
    cal = y[T - n_cal:]
    expected = cal[lag:] - cal[:-lag]          # the old code path

    np.testing.assert_allclose(mdl._residuals_lag(lag), expected)


def test_scaling_when_lag_must_be_shortened():
    """Fallback 3 rescales by sqrt(L/l) rather than silently using a shorter horizon."""
    y = np.arange(10, dtype=float) ** 1.5      # only 10 points
    mdl = ConformalSeasonalPool().fit(y, 1)
    R_short = mdl._residuals_lag(1)
    R_long = mdl._residuals_lag(9)             # cannot be measured directly at lag 9
    assert np.abs(R_long).mean() > np.abs(R_short).mean(), (
        "a longer horizon must not produce smaller residuals")
