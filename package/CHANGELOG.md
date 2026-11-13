# Changelog

All notable changes to `csp-forecaster` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-07-27

### Fixed

- **`_residuals_lag` no longer starves, and no longer returns zero-width intervals.**
  A correctness bug affecting `residual_mode="h_step"` (the default) on **short series**.

  With `h_step` the residual lag grows as `L_h = m * ceil(h / m)`, but the pool was drawn
  from the calibration window only — half the history by default — so the pool *shrank* as
  the horizon grew. On short series it starved completely:

  ```
  m3_yearly:  m=1, H=6, |history|=14  ->  n_cal=7
  residual pool size by horizon:  6, 5, 4, 3, 2, 1
  ```

  A one-element pool makes every predictive draw identical: the distribution collapses to a
  point mass, the 95% interval has **width exactly 0.0**, and **coverage is exactly 0.0**.
  Additionally, when the pool would have been empty the code silently fell back to
  `np.diff(cal)` — *one-step* residuals used to size an H-step-ahead interval, which is far
  too narrow, with no warning.

  The pool is now built with three ordered fallbacks:

  1. calibration window, if it yields at least `MIN_RESIDUALS` (8) residuals —
     **pre-0.2.0 behaviour**, and the branch that always fires on long series;
  2. otherwise the **full history at the same lag** — widen where we look before
     compromising what we measure;
  3. otherwise the longest usable lag `l < L`, rescaled by `sqrt(L / l)` (random-walk
     scaling of multi-step error), and finally `sqrt(L)`-scaled one-step differences.

  Measured on `m3_yearly`, interval width by horizon goes from
  `404, 697, 819, 628, 224, 0.0` to `436, 825, 1072, 1091, 912, 933`, and mean coverage
  from **0.300 to 0.650**.

  **Long-series behaviour is unchanged by construction** — branch 1 always fires when the
  history is long. Verified bit-identical on `electricity`, and statistically
  indistinguishable across six long Tier-B datasets (paired Wilcoxon on
  `|coverage - 0.95|`, p = 0.27, 780 windows).

### Impact — evaluated on 249 dataset-runs / 235 datasets / 24,305 windows

| metric | before | after |
|---|--:|--:|
| mean `\|coverage - 0.95\|` | 0.0715 | **0.0574** |
| mean coverage | 0.8950 | **0.9094** |
| MSIS | 29.78 | **28.58** |
| log predictive density | −5.319 | **−5.172** |
| PIT edge mass | 0.1438 | **0.1304** |
| windows with zero-width intervals | 1.84% | **1.17%** |
| datasets with coverage < 0.30 | 3 | **0** |
| datasets with coverage < 0.50 | 9 | **1** |
| scaled CRPS | **1.573** | 1.605 |

Scaled CRPS is ~2% worse: widening starved intervals costs sharpness. Every other metric
improves, and paired tests against the previous behaviour give p = 1.4e-66 on calibration
and p = 1.8e-12 on MSIS.

### Added

- `ConformalSeasonalPool.MIN_RESIDUALS` (default 8) — class attribute controlling the
  minimum residual-pool size before the widening fallbacks engage.

## [0.1.2] — 2026-06-13

### Fixed

- **`CSPModel._result_dict` now applies the orientation-correct
  finite-sample conformal quantile.** Previously, the `statsforecast`-
  compatible wrapper built `lo-L` / `hi-L` outputs by calling plain
  `np.quantile` on the raw sample array, bypassing the orientation
  correction shipped in v0.1.1 (which applied only to
  `PredictionResult.lower` / `.upper` via `_finalize`). Users
  calling CSP through `CSPModel.predict` / `CSPModel.forecast`
  therefore received the same anti-conservative lower bound v0.1.1 was
  designed to fix. With this change `CSPModel`'s `lo-L` / `hi-L`
  outputs match `ConformalSeasonalPool.predict(...).lower` / `.upper`
  to floating-point precision.

  Benchmarks over a 197-series probabilistic forecasting test set
  showed a 1.7 pp coverage gap at α=0.10 and 1.8 pp at α=0.05 between
  the two paths before the fix; after the fix both paths agree to
  Monte-Carlo tolerance.

### Added

- **Optional Numba JIT acceleration** for the per-step batched quantile
  inside `_finalize`. The pure-numpy path remains the default and the
  only behaviourally tested path; the JIT replacement is a drop-in
  speed optimisation that activates automatically when Numba is
  installed and the environment variable `CSP_NO_NUMBA` is not set to
  `"1"`. Install via `pip install csp-forecaster[numba]`.

  Expected speedup: ~5–15% on per-step time once the JIT cache is warm
  (the first call carries a 2–3 s one-time compile cost cached to
  `__pycache__`). Numpy and Numba paths produce byte-equivalent output;
  Numba is strictly a performance option, not a correctness fix.

- `tests/test_cspmodel_orientation.py` — asserts that
  `CSPModel.predict` returns `lo-L` / `hi-L` matching
  `ConformalSeasonalPool.predict` `lower` / `upper` to floating-point
  precision.

- `tests/test_numba_equivalence.py` — asserts that the Numba-JIT path
  (when installed) produces sample quantiles byte-equivalent to the
  pure-numpy path.

## [0.1.1] — 2026-06-13

### Fixed

- **Orientation-correct finite-sample conformal quantile in `_finalize`.**
  Previously, both the lower and upper interval bounds were computed by a
  single plain `np.quantile(samples, taus, axis=1)` call. Plain
  `np.quantile` uses linear interpolation between order statistics, which
  is *anti-conservative* on the lower tail of a finite-sample empirical
  CDF: it rounds the lower-bound index toward the median, picking a
  less-extreme value than the textbook Romano-style correction would. The
  effect on a typical run with `n_samples=100` and a 95% interval is to
  shift the lower bound roughly half an order statistic *up* (toward the
  median), reducing finite-sample coverage by ~1–2 percentage points
  versus the orientation-corrected version. The same plain-quantile call
  is anti-conservative on neither tail at exactly `0.5` — but the
  asymmetric finite-sample bias is enough to be measurable on real data.

  The fix is a five-line change: a static helper
  `ConformalSeasonalPool._oriented_index(q, n)` that returns
  `floor((n+1)·q)/n` for `q < 0.5` and `ceil((n+1)·q)/n` for `q >= 0.5`,
  followed by remapping the requested `taus` through this helper before
  the (still single, still vectorized) `np.quantile` call. Performance
  cost: zero. The fast path stays vectorized; the legacy path is
  untouched and remains bit-exact under a fixed global seed.

  On a 197-series Monash benchmark (T_train=800, T_test=300, alpha-grid
  {0.02, 0.05, 0.1, 0.2, 0.5}, primary `alpha=0.10`) the fix closes the
  coverage gap to an independent reference implementation
  (`conformal_ts.CSP`) from ~1.7 pp to within Monte-Carlo tolerance, and
  reduces both per-side miss rates (paired Wilcoxon p < 1e-12 on
  `cov_a10` and `right_miss`, p < 1e-2 on `left_miss` and `wis`).

### Added

- `tests/test_orientation_correction.py` — 8 regression tests covering
  the `_oriented_index` rule itself, the lower-bound-is-no-less-extreme
  invariant after the patch, and a 20-seed coverage sanity check on a
  stationary synthetic series.

### Verified

- `tests/test_fast_equivalence.py` — fast vs legacy still agree on CRPS,
  coverage, and sample-moment metrics (orientation correction is applied
  in `_finalize`, which both modes call).
- `tests/test_legacy_golden.py` — legacy mode is still bit-exact
  deterministic under a fixed global NumPy RNG seed.

### Background

The orientation correction was identified during a side-by-side
comparison with an independent CSP implementation that already applied
the textbook finite-sample rule. The 1.7 pp coverage gap traced cleanly
to this single line in `_finalize`. The fix is consistent with the
Romano–Patterson–Candès (NeurIPS 2019) finite-sample correction for
one-sided conformal scores, applied symmetrically across both tails of a
two-sided empirical predictive distribution.

## [0.1.0] — 2026-06-08

### Added

- Initial release of `csp-forecaster`.
- `ConformalSeasonalPool` core algorithm with two execution paths:
  - `mode="legacy"` — per-horizon Python loop using NumPy's global RNG;
    reproduces the published paper sample-for-sample.
  - `mode="fast"` — vectorized draws using an explicit seeded
    `Generator`; roughly 1.2× faster than `legacy` on one-shot prediction.
- `csp_forecaster.nixtla.CSPModel` — `statsforecast`-compatible adapter.
- Test suite covering fast/legacy equivalence on CRPS and coverage,
  legacy reproducibility under a fixed seed, and the statsforecast
  adapter.
