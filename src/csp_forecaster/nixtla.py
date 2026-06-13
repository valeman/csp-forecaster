"""
csp_forecaster.nixtla
=====================

A thin adapter that exposes Conformal Seasonal Pools through the Nixtla
``statsforecast`` per-model protocol, so CSP drops directly into
``StatsForecast(models=[...])``.

The statsforecast model protocol used here:

    model = CSPModel(season_length=24)
    model.fit(y)                          # y: 1-D np.ndarray
    model.predict(h, level=[80, 95])      # -> {'mean', 'lo-80','hi-80','lo-95','hi-95'}
    model.forecast(y, h, level=[95])      # one-shot fit+predict, same dict shape

and through the orchestrator (long dataframe with columns unique_id, ds, y):

    from statsforecast import StatsForecast
    from csp_forecaster.nixtla import CSPModel
    sf = StatsForecast(models=[CSPModel(season_length=24, alias="CSP")], freq="H")
    sf.fit(df)
    sf.predict(h=24, level=[95])          # columns: CSP, CSP-lo-95, CSP-hi-95

``statsforecast`` itself is NOT a dependency of this module: the adapter only
*conforms* to the protocol (it is pure NumPy). Install statsforecast only if you
want to use the orchestrator.

Level mapping: a confidence ``level`` of L (percent) maps to ``alpha = 1 - L/100``;
the returned ``lo-L`` / ``hi-L`` are the ``alpha/2`` and ``1 - alpha/2`` empirical
quantiles of the CSP predictive samples. ``mean`` is the sample mean per horizon.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .core import ConformalSeasonalPool


class CSPModel:
    """statsforecast-compatible wrapper around :class:`ConformalSeasonalPool`."""

    def __init__(
        self,
        season_length: int = 1,
        pool_weight: float = 0.5,
        exp_lambda: float = 0.01,
        cal_fraction: float = 0.5,
        adaptive: bool = True,
        mode: str = "fast",
        n_samples: int = 100,
        alias: str = "CSP",
        random_state: "int | None" = None,
    ):
        self.season_length = season_length
        self.pool_weight = pool_weight
        self.exp_lambda = exp_lambda
        self.cal_fraction = cal_fraction
        self.adaptive = adaptive
        self.mode = mode
        self.n_samples = n_samples
        self.alias = alias
        self.random_state = random_state

    # statsforecast clones models per series; provide a fresh, UNFITTED copy
    # built from the constructor parameters (fit state is intentionally dropped).
    def new(self) -> "CSPModel":
        return CSPModel(
            season_length=self.season_length,
            pool_weight=self.pool_weight,
            exp_lambda=self.exp_lambda,
            cal_fraction=self.cal_fraction,
            adaptive=self.adaptive,
            mode=self.mode,
            n_samples=self.n_samples,
            alias=self.alias,
            random_state=self.random_state,
        )

    def __repr__(self) -> str:
        return self.alias

    # ------------------------------------------------------------------ core
    def _make(self) -> ConformalSeasonalPool:
        return ConformalSeasonalPool(
            pool_weight=self.pool_weight,
            exp_lambda=self.exp_lambda,
            cal_fraction=self.cal_fraction,
            adaptive=self.adaptive,
            mode=self.mode,
            random_state=self.random_state,
        )

    def _result_dict(self, samples: np.ndarray, level: Optional[List[int]]) -> Dict[str, np.ndarray]:
        """Build the statsforecast-style output dict from a sample matrix.

        Per-level quantiles are routed through
        :meth:`ConformalSeasonalPool._oriented_index` so the ``lo-L`` /
        ``hi-L`` bounds receive the same orientation-correct finite-
        sample correction the core ``predict`` path applies in
        ``_finalize``. Users hitting CSP via
        ``StatsForecast(models=[CSPModel(...)])`` therefore see exactly
        the same bounds as users calling
        ``ConformalSeasonalPool.predict(...)`` directly.

        Prior to v0.1.2 this used plain ``np.quantile``, which was
        anti-conservative on the lower tail and reproduced the same bug
        v0.1.1 fixed in the core class. The two paths now agree to
        floating-point precision.
        """
        n = samples.shape[1]
        out: Dict[str, np.ndarray] = {"mean": samples.mean(axis=1)}
        if level:
            for lv in level:
                a = 1.0 - lv / 100.0
                q_lo = ConformalSeasonalPool._oriented_index(a / 2.0, n)
                q_hi = ConformalSeasonalPool._oriented_index(1.0 - a / 2.0, n)
                lo, hi = np.quantile(samples, [q_lo, q_hi], axis=1)
                out[f"lo-{lv}"] = lo
                out[f"hi-{lv}"] = hi
        return out

    # --------------------------------------------------------- protocol API
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "CSPModel":
        self.model_ = self._make().fit(np.asarray(y, dtype=float).ravel(), self.season_length)
        return self

    def predict(
        self,
        h: int,
        X: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        if not hasattr(self, "model_"):
            raise RuntimeError("call fit() before predict()")
        res = self.model_.predict(h, alpha=0.05, n_samples=self.n_samples)
        return self._result_dict(res.samples, level)

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, np.ndarray]:
        """One-shot fit-then-predict (the method StatsForecast.forecast calls)."""
        model = self._make().fit(np.asarray(y, dtype=float).ravel(), self.season_length)
        res = model.predict(h, alpha=0.05, n_samples=self.n_samples)
        return self._result_dict(res.samples, level)
