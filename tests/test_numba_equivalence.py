"""Regression tests for the optional Numba JIT path added in v0.1.2.

The Numba kernel in ``csp_forecaster.core._batched_quantile`` is a
strict performance optimisation: it must return values byte-equivalent
to the pure-numpy ``np.quantile(samples, oriented_taus, axis=1)`` path.

Two scenarios:

1. Numba is NOT installed (CI default, end-user default install). The
   module-level ``_USE_NUMBA`` flag is False and ``_batched_quantile``
   resolves to the numpy fallback. The equivalence assertion is trivial
   (the function IS np.quantile under a wrapper) and we just exercise
   the call path.

2. Numba IS installed. ``_USE_NUMBA`` is True and ``_batched_quantile``
   is the JIT'd kernel. We compare its output against a freshly-computed
   ``np.quantile(samples, oriented_taus, axis=1)`` on the same input
   and assert ``allclose(rtol=1e-12)``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csp_forecaster.core import _USE_NUMBA, _batched_quantile   # noqa: E402


def test_batched_quantile_dispatch_resolvable():
    """The dispatching helper exists and is callable regardless of
    whether Numba is installed."""
    samples = np.random.default_rng(0).normal(size=(8, 100))
    taus = np.array([0.025, 0.5, 0.975])
    out = _batched_quantile(samples, taus)
    assert out.shape == (3, 8)
    assert np.all(np.isfinite(out))


def test_batched_quantile_matches_np_quantile():
    """Whichever path is active, the output must agree with
    ``np.quantile(samples, taus, axis=1)`` within floating-point
    tolerance.
    """
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(16, 100))
    taus = np.array([0.005, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.995])
    out = _batched_quantile(samples, taus)
    expected = np.quantile(samples, taus, axis=1)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_batched_quantile_handles_extreme_taus():
    """Tail edges (q very close to 0 or 1) must be safe."""
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(4, 50))
    taus = np.array([1e-9, 0.001, 0.999, 1 - 1e-9])
    out = _batched_quantile(samples, taus)
    assert out.shape == (4, 4)
    assert np.all(np.isfinite(out))
    # Lower-tail quantile <= upper-tail quantile per horizon
    assert np.all(out[0] <= out[3] + 1e-9)


def test_batched_quantile_jit_flag_consistent():
    """If _USE_NUMBA is True, the kernel must be a JIT'd function (not
    the pure-numpy fallback). If False, it must be the numpy fallback.
    Catches accidental import-order regressions."""
    if _USE_NUMBA:
        # Numba-decorated functions have a `.py_func` attribute.
        assert hasattr(_batched_quantile, "py_func") or hasattr(
            _batched_quantile, "__wrapped__"
        ), (
            "_USE_NUMBA is True but _batched_quantile does not look "
            "like a JIT'd function."
        )
    else:
        # Pure-numpy version is a plain Python function.
        assert _batched_quantile.__name__ == "_batched_quantile"
