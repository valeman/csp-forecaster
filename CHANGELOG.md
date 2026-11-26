# Changelog

All notable changes to `csp-forecaster` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-11-26

### Added

- **`CSPAuto`: CSP that chooses its own set-up on the history.** Two things are chosen
  together, by weighted interval score on the last fifth of the history (at most 1,000
  points), one step ahead by default:

  - *the target*: the levels (plain CSP), or the one-step changes, anchored on the last
    observed value. With changes, CSP forecasts `d_t = y_t - y_{t-1}` and every sample path
    is the last value plus the cumulative sampled changes;
  - *the seasonal period*: none, the declared period, and the hour for sub-hourly data
    declared with a daily period (`auto_candidates`: 288 tries 1, 12, 288; 1440 tries 1, 60,
    1440). With `acf_peak=True` the highest autocorrelation peak of the history is tried too
    (lag 2 or more, at least 10 cycles, autocorrelation at least 0.3; `acf_peak()`): the short
    cycles of fast machine and sensor data.

  ```python
  from csp_forecaster import CSPAuto
  model = CSPAuto(random_state=0).fit(history, seasonal_period=288)
  r = model.predict(H=1, alpha=0.05)
  model.choice_          # target, period, declared period, all scores
  ```

  Nothing in `ConformalSeasonalPool` changes: every existing call gives the same output.

  This is the method the large conformal forecasting benchmark runs as `CSP_auto` (and, with
  `acf_peak=True`, `CSP_auto_acf`). Measured there, one step ahead, weighted interval score
  per dataset against CSP with the declared period (geometric mean over about 2,900
  datasets): 0.47, better on 1,964 datasets and worse on 158; datasets with 95% coverage
  under 0.90 fall from 26% to 10%. The gain is largest on 1-minute and 5-minute data and
  absent on 10-second and yearly data. The autocorrelation-peak period adds a further 7% on a
  fresh sample of 115 series (0.928, 95% interval 0.858 to 0.980), almost all of it on
  10-second and 1-minute data.

  Checked against the benchmark's own implementation on 40 real series (10-second to daily):
  the same target and period on 90% of them, weighted interval score 0.989 of the
  benchmark's (median 0.997); with `acf_peak=True`, 83% and 1.002 on 24 series. The
  differences come from scoring 200 evenly spaced origins rather than every point of the tail
  (`max_origins=None` scores every point) and from random draws.

## [0.2.3] — 2026-11-22

### Changed

- **Short-history guard, on by default.** `fit()` uses no seasonal period when the
  history is shorter than one seasonal period (`short_history_guard=True`). A season
  that has not been observed once offers no earlier value at the same phase to pool
  or to anchor on; CSP nevertheless centred on such phases and kept the interval
  width flat across steps. The period passed to `fit` is kept in
  `seasonal_period_given_`; `short_history_guard_active_` says whether the guard
  fired. Not applied in `mode="legacy"`. `short_history_guard=False` restores the
  previous behaviour. `CSPModel` takes the same argument.

  Whenever the history holds at least one full period, forecasts are identical to
  0.2.2 bit for bit (tested).

  Measured, 95% intervals:

  - *CSP benchmark, 24,502 windows:* the guard fires on 3 windows (monthly series
    with 10 points). Two get better (interval score 948 to 706), one gets worse
    (814 to 1,343). Everything else is unchanged.
  - *Benchmark series with a declared period of 48 or more, history cut to half a
    period* (1,346 windows, 43 datasets): interval score 24% lower and CRPS 23%
    lower (geometric mean over datasets); better on 28 datasets and worse on 15
    for the interval score, 33 and 10 for CRPS. The gain is largest on 5-minute
    data (median ratio 0.42) and absent on half-hourly and 15-minute data (about
    1.0). Coverage rises from 0.38 to 0.50: half a period of history remains a
    hard case with or without the guard.
  - *One-minute data with a daily period and 720 to 816 points of history* (three
    datasets, 18 windows each, measured 15 Nov 2026): interval score 191 to 29,
    248 to 11 and 141 to 11; coverage 39%, 36% and 73% to 92%, 88% and 98%.

## [0.2.2] — 2026-11-21

### Added

- **`seasonal_period="auto"`: the seasonal period is chosen on the history.** Opt-in;
  an integer period behaves exactly as before (bit-identical, tested).

  ```python
  csp.fit(history, "auto", freq="5min")                 # tries none, hour, day, week
  csp.fit(history, "auto", freq="H", candidates=[12])   # adds a period of your own
  csp.fit(history, "auto", candidates=[1, 60, 1440])    # your own list only
  csp.fit(history, "auto", freq="H", horizon=24)        # chosen for a 24-step forecast
  ```

  By default the periods tried are the natural cycles of the sampling frequency
  (`seasonal_candidates(freq)`; 1 = no season is always among them). Each is scored by
  weighted interval score on the last fifth of the history (at most 1,000 points and
  200 forecast origins) at the given horizon; the lowest wins, ties go to the shorter
  period. Only the history is used. Scores are kept in `season_scores_`. The choice
  uses a private fixed seed and does not change the model's random draws.

  Why: with `m > 1` every draw is built from values one or more seasons back and the
  last observed value is not used. A long declared period (1440 for minute data) is
  therefore only right when the series repeats closely from season to season.

  Where it helps and where it does not (measured 20–21 Nov 2026):

  - *One step ahead, long history, high-frequency data* (190 series, 5,000 points of
    history): median weighted interval score falls by 58% on 1-minute, 65% on 5-minute
    and 49% on series with periods 7 to 168; coverage at 95% goes from 0.88 / 0.90 /
    0.93 to 0.94 / 0.95 / 0.93.
  - *The CSP benchmark's Laplace subset* (2,534 windows, 228 datasets, histories cut to
    400–720 points, horizons 4 to 168): no gain. Against the declared period it wins
    CRPS on 50 datasets and loses on 45 (133 equal, p = 0.29), and calibration is
    slightly worse (mean coverage 0.894 against 0.898). With so little history the
    validation tail is 80 to 144 points and the choice is noisy. Keep the declared
    period in that setting.

- `seasonal_candidates(freq)`: the default candidate periods for a frequency string.
- `CSPModel(season_length="auto", freq="H")` and `season_candidates=[...]`.

## [0.2.1] — 2026-11-20

### Fixed

- **`mode="fast"` (the default) now returns float64 samples; it returned float32.**
  The history is float64 and `mode="legacy"` always sampled in float64, but the fast
  path allocated its sample matrix as float32, so every draw — and therefore every
  interval bound and quantile — was rounded to about seven significant digits. Two
  consequences, both reproduced on v0.2.0:

  - **False misses on flat and exactly periodic series.** When every draw is the same
    value and that value is not exactly representable in float32 (`0.1`, `1234567.891`),
    the degenerate interval sat on the *rounded* value and excluded the true one by
    ~1e-9: empirical coverage 0.0 instead of 1.0. Values exact in float32 (`0.0`, `5.0`)
    were unaffected.
  - **Interval collapse on large-level, low-noise series.** Once the level exceeds
    roughly 1e7 times the noise scale, float32 cannot represent the predictive spread
    at all (its spacing is 8 at 1e8 and 64 at 1e9), so the interval collapsed to a
    single point — a real loss of coverage, not a rounding artefact.

  Both modes now sample in float64. Memory for the sample matrix doubles
  (`H × n_samples × 8` bytes — 77 kB at H=48, 200 samples). On series at ordinary
  scales, bounds move only in the seventh significant digit.

### Changed

- `test_fast_samples_are_float32` (which pinned the old dtype) is replaced by
  `test_fast_samples_are_float64`; `tests/test_float_precision.py` adds regression
  tests for the flat, exactly periodic and large-level cases in both modes.

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
