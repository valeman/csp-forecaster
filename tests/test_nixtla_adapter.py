"""Tests for the statsforecast-compatible CSPModel adapter.

The protocol tests use NumPy only. An end-to-end test through the real
``StatsForecast`` orchestrator runs only if ``statsforecast`` is installed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csp_forecaster import CSPModel   # noqa: E402


def _series(seed=0, n=400, m=24):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / m) + 0.05 * t + rng.normal(0, 0.5, n)


def test_predict_returns_protocol_keys():
    y = _series()
    m = CSPModel(season_length=24, mode="fast", random_state=0).fit(y)
    out = m.predict(h=24, level=[80, 95])
    assert set(out) == {"mean", "lo-80", "hi-80", "lo-95", "hi-95"}
    for k, v in out.items():
        assert v.shape == (24,)
    # Intervals are ordered and nested.
    assert np.all(out["lo-95"] <= out["lo-80"])
    assert np.all(out["hi-80"] <= out["hi-95"])
    assert np.all(out["lo-95"] <= out["hi-95"])


def test_predict_without_level_is_point_only():
    m = CSPModel(season_length=24).fit(_series())
    out = m.predict(h=12)
    assert set(out) == {"mean"}
    assert out["mean"].shape == (12,)


def test_forecast_one_shot_matches_shapes():
    y = _series()
    out = CSPModel(season_length=24, random_state=1).forecast(y, h=24, level=[95])
    assert set(out) == {"mean", "lo-95", "hi-95"}
    assert out["mean"].shape == (24,)


def test_new_returns_independent_unfitted_copy():
    m = CSPModel(season_length=7, alias="CSP-7").fit(_series(m=7))
    clone = m.new()
    assert clone.alias == "CSP-7" and clone.season_length == 7
    assert not hasattr(clone, "model_")   # fresh, unfitted


@pytest.mark.skipif(
    pytest.importorskip("statsforecast", reason="statsforecast not installed") is None,
    reason="statsforecast not installed",
)
def test_runs_inside_statsforecast_orchestrator():
    import pandas as pd
    from statsforecast import StatsForecast

    n = 240
    df = pd.DataFrame({
        "unique_id": "s1",
        "ds": np.arange(n),
        "y": _series(n=n, m=24),
    })
    sf = StatsForecast(models=[CSPModel(season_length=24, alias="CSP", random_state=0)], freq=1)
    fc = sf.forecast(df=df, h=24, level=[95])
    assert "CSP" in fc.columns
    assert "CSP-lo-95" in fc.columns and "CSP-hi-95" in fc.columns
    assert len(fc) == 24
