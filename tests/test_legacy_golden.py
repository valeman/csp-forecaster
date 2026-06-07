"""Golden test: the packaged legacy path is bit-exact with the original paper code.

Under an identical global NumPy seed and identical fitted history, the packaged
``ConformalSeasonalPool(mode="legacy")`` must produce sample arrays that are
element-for-element equal to the original
``cp_bench.methods.ConformalSeasonalPool``.

Run from the repo root:
    PYTHONPATH=src:csp_forecaster/src python -m pytest csp_forecaster/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))                       # original cp_bench
sys.path.insert(0, str(ROOT / "csp_forecaster" / "src"))   # packaged version

from csp_forecaster import ConformalSeasonalPool as Packaged   # noqa: E402

try:
    from cp_bench.methods import ConformalSeasonalPool as Original   # noqa: E402
    HAVE_ORIGINAL = True
except Exception:
    HAVE_ORIGINAL = False


def _series(seed, n=400, m=24):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (10 + 3 * np.sin(2 * np.pi * t / m) + 0.05 * t + rng.normal(0, 0.5, n))


@pytest.mark.skipif(not HAVE_ORIGINAL, reason="original cp_bench not importable")
@pytest.mark.parametrize("adaptive", [True, False])
@pytest.mark.parametrize("m,H", [(24, 24), (7, 14), (1, 10)])
def test_legacy_bit_exact(adaptive, m, H):
    hist = _series(0, m=max(m, 2))
    orig = Original(adaptive=adaptive).fit(hist, seasonal_period=m)
    pkg = Packaged(adaptive=adaptive, mode="legacy").fit(hist, seasonal_period=m)

    np.random.seed(12345)
    r_orig = orig.predict(H, n_samples=100)
    np.random.seed(12345)
    r_pkg = pkg.predict(H, n_samples=100)

    np.testing.assert_array_equal(r_orig.samples, r_pkg.samples)
    for tau in r_orig.quantiles:
        np.testing.assert_array_equal(r_orig.quantiles[tau], r_pkg.quantiles[tau])
