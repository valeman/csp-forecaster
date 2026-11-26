"""
csp_forecaster.auto
===================

``CSPAuto``: CSP that chooses its own set-up on the history (added in 0.3.0).

Two things are chosen together, by weighted interval score on the last part of the
history, one step ahead:

* **the target**: the levels ``y_t`` (plain CSP), or the one-step changes
  ``d_t = y_t - y_{t-1}``. With changes, CSP forecasts ``d`` and the forecast of ``y`` is
  the last observed value plus the forecast change. On series that wander (random walks,
  level shifts, slow drifts) this anchors the forecast on the last value, where plain CSP
  with a long season anchors it on the value one season ago;
* **the seasonal period**: no season (1), the declared period ``m``, and, for sub-hourly
  data whose declared period is a day (``m`` a multiple of 24 and at least 96), the hour
  (``m // 24``). With ``acf_peak=True`` the highest autocorrelation peak of the history is
  tried as well (lag 2 or more, at least 10 full cycles, autocorrelation at least 0.3):
  this finds the short cycles (3 to 10 steps) of fast machine and sensor data that a
  declared daily period misses.

This is the method the large benchmark runs as ``CSP_auto`` (and, with
``acf_peak=True``, as ``CSP_auto_acf``); see the changelog for the measured effect.

Use::

    from csp_forecaster import CSPAuto
    model = CSPAuto(random_state=0).fit(history, seasonal_period=288)   # declared period
    r = model.predict(H=1, alpha=0.05)
    model.choice_      # {'target': 'change', 'seasonal_period': 1, 'scores': {...}, ...}

What is chosen depends only on the history passed to ``fit``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .core import DEFAULT_QUANTILE_LEVELS, ConformalSeasonalPool, PredictionResult

#: Central-interval levels of the weighted interval score used for the choice.
SELECT_ALPHAS: Tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TARGETS = ("level", "change")


def auto_candidates(m: int) -> List[int]:
    """Seasonal periods ``CSPAuto`` tries for a declared period ``m``.

    No season, ``m`` itself, and the hour of a sub-hourly series declared with a daily
    period (``m // 24`` when ``m`` is a multiple of 24 and ``m // 24 >= 4``: 288 -> 12,
    1440 -> 60, 96 -> 4).
    """
    m = max(1, int(m))
    c = {1, m}
    if m >= 96 and m % 24 == 0 and m // 24 >= 4:
        c.add(m // 24)
    return sorted(c)


def acf_peak(y: ArrayLike, lo: int = 2, min_cycles: int = 10, thr: float = 0.3,
             nmax: int = 5000) -> Optional[int]:
    """Lag of the highest autocorrelation peak, or ``None``.

    Searched between ``lo`` and ``len(y) // min_cycles`` (at most 1,440) on the last ``nmax``
    finite values; a peak is a lag whose autocorrelation is at least that of both
    neighbours. Returned only when its autocorrelation is at least ``thr``.
    """
    y = np.asarray(y, float)[-nmax:]
    y = y[np.isfinite(y)]
    n = len(y)
    hi = min(1440, n // min_cycles)
    if hi < lo + 1 or np.std(y) == 0:
        return None
    z = y - y.mean()
    f = np.fft.rfft(z, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:hi + 2]
    a = a / a[0]
    k = np.arange(lo, hi + 1)
    peaks = [j for j in k if a[j] >= a[j - 1] and a[j] >= a[j + 1]]
    if not peaks:
        return None
    j = max(peaks, key=lambda j: a[j])
    return int(j) if a[j] >= thr else None


def _wis(actual: np.ndarray, q: Dict[float, np.ndarray], alphas: Sequence[float]) -> float:
    """Weighted interval score (central intervals ``alphas`` plus the median term)."""
    s = 0.5 * np.abs(actual - q[0.5])
    for a in alphas:
        lo, hi = q[a / 2], q[1 - a / 2]
        s = s + a / 2 * (hi - lo) + np.maximum(lo - actual, 0) + np.maximum(actual - hi, 0)
    return float(np.mean(s)) / (len(alphas) + 0.5)


class CSPAuto:
    """Conformal Seasonal Pools with the target and the seasonal period chosen on the history.

    Parameters
    ----------
    acf_peak : bool, default False
        Also try the highest autocorrelation peak of the history as a seasonal period.
    targets : sequence of {"level", "change"}, default both
        Targets to choose between. ``("level",)`` reduces ``CSPAuto`` to a choice of the
        seasonal period only.
    val_fraction, max_val : float, int
        The choice is scored on the last ``val_fraction`` of the history, at most
        ``max_val`` points (defaults 0.2 and 1,000, as in the benchmark).
    max_origins : int or None, default 200
        Forecast origins scored in that tail, evenly spaced. ``None`` scores every point,
        as the benchmark does (slower: one CSP forecast per point, target and period).
    min_val : int, default 20
        With fewer validation points than this the history is too short to choose:
        ``CSPAuto`` falls back to plain CSP with the declared period.
    n_samples_select : int, default 200
        Predictive samples per trial forecast in the choice.
    **csp_kwargs
        Options of :class:`ConformalSeasonalPool` (``pool_weight``, ``exp_lambda``,
        ``residual_mode``, ``random_state`` ...), used for the trial forecasts and for the
        final model.

    After ``fit``: ``target_`` ("level" or "change"), ``seasonal_period_`` (chosen),
    ``choice_`` (target, period, declared period, validation points, all scores) and
    ``model_`` (the fitted :class:`ConformalSeasonalPool`).
    """

    def __init__(self, acf_peak: bool = False, targets: Sequence[str] = TARGETS,
                 val_fraction: float = 0.2, max_val: int = 1000,
                 max_origins: Optional[int] = 200, min_val: int = 20,
                 n_samples_select: int = 200, **csp_kwargs: Any):
        bad = [t for t in targets if t not in TARGETS]
        if bad or not targets:
            raise ValueError(f"targets must be a non-empty subset of {TARGETS}, got {tuple(targets)!r}")
        if csp_kwargs.get("mode", "fast") != "fast":
            raise ValueError("CSPAuto uses mode='fast'")
        self.acf_peak = bool(acf_peak)
        self.targets = tuple(targets)
        self.val_fraction = float(val_fraction)
        self.max_val = int(max_val)
        self.max_origins = None if max_origins is None else int(max_origins)
        self.min_val = int(min_val)
        self.n_samples_select = int(n_samples_select)
        self.csp_kwargs = dict(csp_kwargs)
        self.history: Optional[np.ndarray] = None
        self.target_: str = "level"
        self.seasonal_period_: int = 1
        self.choice_: Dict[str, Any] = {}
        self.model_: Optional[ConformalSeasonalPool] = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _series(y: np.ndarray, target: str) -> np.ndarray:
        return y if target == "level" else np.diff(y)

    def _trial(self) -> ConformalSeasonalPool:
        kw = {k: v for k, v in self.csp_kwargs.items() if k not in ("random_state", "mode")}
        return ConformalSeasonalPool(mode="fast", random_state=0, **kw)

    def _score(self, y: np.ndarray, target: str, c: int, origins: np.ndarray, H: int) -> float:
        """Mean weighted interval score of (target, period) over the origins, on the levels."""
        trial = self._trial()
        levels = sorted({0.5, *[a / 2 for a in SELECT_ALPHAS], *[1 - a / 2 for a in SELECT_ALPHAS]})
        total = 0.0
        for t0 in origins:
            past = y[:t0]
            trial.fit(self._series(past, target), c)
            r = trial._predict_fast(H, 0.05, levels, self.n_samples_select)
            if target == "level":
                q = r.quantiles
            else:                                # forecast of y = last value + cumulative change
                S = past[-1] + np.cumsum(r.samples, axis=0)
                q = {t: np.quantile(S, t, axis=1) for t in levels}
            total += _wis(y[t0:t0 + H], q, SELECT_ALPHAS)
        return total / len(origins)

    # ---------------------------------------------------------------------- fit
    def fit(self, history: ArrayLike, seasonal_period: int = 1, horizon: int = 1) -> "CSPAuto":
        """Choose the target and the seasonal period on ``history``, then fit.

        ``seasonal_period`` is the declared period of the series (the season length of
        its frequency: 24 for hourly, 288 for 5-minute, 1440 for minute data, 1 if none).
        ``horizon`` is the forecast horizon the choice is made for.
        """
        y = np.asarray(history, dtype=np.float64).ravel()
        self.history = y
        T = len(y)
        m = max(1, int(seasonal_period))
        H = max(1, int(horizon))
        v = int(min(self.max_val, T * self.val_fraction))
        first = T - v
        cands = auto_candidates(m)
        peak = None
        if self.acf_peak and first > 0:
            peak = acf_peak(y[:first])
            if peak:
                cands = sorted(set(cands) | {peak})
        usable = [c for c in cands if c <= first // 2]
        targets = [t for t in self.targets if t == "level" or T >= 3]
        scores: Dict[Tuple[str, int], float] = {}
        origins = np.arange(max(first, 1), T - H + 1)
        if self.max_origins is not None and len(origins) > self.max_origins:
            origins = origins[np.linspace(0, len(origins) - 1, self.max_origins).round().astype(int)]
        if v >= self.min_val and len(origins) and usable:
            for t in targets:
                for c in usable:
                    try:
                        scores[(t, c)] = self._score(y, t, c, origins, H)
                    except Exception:            # a candidate that cannot run is not chosen
                        scores[(t, c)] = np.inf
        if scores and np.isfinite(min(scores.values())):
            # ties: shorter period first, then levels (plain CSP) before changes
            target, season = min(scores, key=lambda k: (scores[k], k[1], k[0] != "level"))
        else:
            target, season = ("level" if "level" in self.targets else self.targets[0]), m
        self.target_, self.seasonal_period_ = target, int(season)
        self.choice_ = {"target": target, "seasonal_period": int(season), "declared": m,
                        "acf_peak": peak, "val_points": v, "origins": int(len(origins)),
                        "scores": {f"{k[0]}_{k[1]}": float(s) for k, s in scores.items()}}
        self.model_ = ConformalSeasonalPool(**self.csp_kwargs).fit(self._series(y, target), int(season))
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, H: int, alpha: float = 0.05, quantile_levels: Optional[List[float]] = None,
                n_samples: int = 100) -> PredictionResult:
        """Forecast ``H`` steps. With the target "change", every sample path is the last
        observed value plus the cumulative sum of the sampled changes."""
        if self.model_ is None or self.history is None:
            raise RuntimeError("call fit() before predict()")
        if quantile_levels is None:
            quantile_levels = DEFAULT_QUANTILE_LEVELS
        r = self.model_.predict(H, alpha=alpha, quantile_levels=quantile_levels, n_samples=n_samples)
        if self.target_ == "level":
            r.method = f"CSP_auto(level, m={self.seasonal_period_})"
            r.metadata["choice"] = dict(self.choice_)
            return r
        S = self.history[-1] + np.cumsum(r.samples, axis=0)
        out = self.model_._finalize(S, alpha, quantile_levels, r.metadata.get("pool_weight_eff", 0.0),
                                    np.zeros(int(r.metadata.get("residual_pool_size", 1))))
        out.method = f"CSP_auto(change, m={self.seasonal_period_})"
        out.metadata["choice"] = dict(self.choice_)
        return out
