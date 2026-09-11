# csp_forecaster

A standalone, installable package for Conformal Seasonal Pools (CSP) — the
training-free probabilistic time-series forecaster from the paper [Training-Free
Probabilistic Time-Series Forecasting with Conformal Seasonal Pools](https://arxiv.org/abs/2605.03789).

CSP produces a full predictive sample per horizon by blending two components:

1. a **seasonal pool** — same-phase historical values weighted by exponential recency, and
2. a **conformal residual** component — a seasonal-naive point forecast plus signed calibration residuals.

No training, no neural network, no learned parameters.

## Publication

The CSP paper has been **accepted at COPA 2026** (Symposium on Conformal and Probabilistic
Prediction with Applications).

A companion article for practitioners is **forthcoming in *Foresight: The International Journal of
Applied Forecasting***.

> Manokhin, V. (2026). *Training-Free Probabilistic Time-Series Forecasting with Conformal
> Seasonal Pools.* Proceedings of Machine Learning Research, vol. 329 (COPA 2026).

Please cite the paper if you use this package. See [`CITATION`](#citation) below.

## Benchmark highlight — a training-free method that competes with tuned classical models

On the classic **AirPassengers** series, in a rolling-origin backtest against the tuned models from
*Mastering Modern Time Series Forecasting* by V. Manokhin
([Pro edition](https://valeman.gumroad.com/l/MasteringModernTimeSeriesForecastingPro) ·
[Standard](https://valeman.gumroad.com/l/MasteringModernTimeSeriesForecasting) ·
[Amazon](https://www.amazon.com/Mastering-Modern-Time-Forecasting-Comprehensive/dp/1919465839)):

| Method | CRPS (H=12) | RMSE (H=12) | CRPS (H=24) | RMSE (H=24) |
|---|--:|--:|--:|--:|
| ETS(M,A,M) — *tuned* | 11.3 | 18.9 | 15.1 | 24.7 |
| **CSP — *training-free*** | **12.8** | 21.2 | **17.4** | 28.0 |
| SARIMA — *tuned* | 13.4 | 24.0 | 20.2 | 37.9 |
| Seasonal-Naive | 13.3 | 42.9 | 25.9 | 62.1 |

1. **CSP beats the tuned SARIMA on both accuracy and full-distribution quality, at every horizon.**
   CRPS 12.8 vs 13.4 (H=12) and 17.4 vs 20.2 (H=24); RMSE 21.2 vs 24.0 and 28.0 vs 37.9. With no
   fitting, no order selection, and no hyper-parameters.
2. **It transforms its own baseline.** Versus plain Seasonal-Naive, CSP roughly halves the point
   error (21.2 vs 42.9; 28.0 vs 62.1) and clearly improves CRPS — the conformal-residual structure
   adds real value.
3. **CSP ranks 2nd of four on CRPS at both horizons** — behind only a correctly-specified, fitted
   ETS(M,A,M) (you must pick the multiplicative model *and* fit it). It produces a full predictive
   distribution in milliseconds.
4. **And this is a hard case for CSP.** AirPassengers is strongly trending — outside CSP's
   stable-level seasonal design domain, where the paper shows it beats the neural DeepNPTS baseline.

Reproducible notebook: [`examples/airpassengers_intervals_comparison.ipynb`](examples/airpassengers_intervals_comparison.ipynb).

## Benchmark — v0.2.0 on 218 datasets

Every method evaluated on **identical rolling-origin windows with matched seeds**:
218 datasets, 231 dataset-runs, **22,822 windows**, drawn from the Monash archive, the GluonTS
and LOTSA collections, the chronos parquet corpus and assorted public sources, at frequencies
from 5-minutely to yearly.

Ranks are per-window over the five methods shown (1 = best). Coverage targets 0.95.
MSIS is the mean scaled interval score; log density and PIT edge measure the full predictive
distribution rather than just the interval.

| method | CRPS rank | Winkler rank | MSIS | coverage | \|cov−0.95\| | log dens | PIT edge |
|---|--:|--:|--:|--:|--:|--:|--:|
| **CSP v0.2.0 (this package)** | **2.32** | **2.30** | **29.1** | **0.910** | **0.057** | **−5.23** | **0.131** |
| Nixtla `ConformalSeasonalPool` | 2.52 | 2.68 | 37.5 | 0.865 | 0.098 | −5.59 | 0.186 |
| DeepNPTS (GluonTS, trained) | 3.11 | 3.95 | 51.8 | 0.566 | 0.384 | −8.40 | 0.481 |
| SeasonalNPTS (GluonTS) | 3.27 | 3.26 | 45.9 | 0.756 | 0.200 | −5.98 | 0.292 |
| NPTS (GluonTS) | 3.78 | 2.80 | 44.1 | 0.847 | 0.111 | −5.63 | 0.207 |

**CSP v0.2.0 ranks first on all seven metrics.**

### Head-to-head, datasets won (of 231)

| vs | CRPS | Winkler | MSIS |
|---|--:|--:|--:|
| Nixtla CSP | 137 | 149 | 154 |
| DeepNPTS | 160 | **185** | **197** |
| SeasonalNPTS | 148 | 152 | 150 |
| NPTS | **174** | 137 | 138 |

Paired Wilcoxon over all 22,822 windows, on calibration (`|coverage − 0.95|`):

| vs | CSP v0.2.0 | other | *p* |
|---|--:|--:|--:|
| Nixtla CSP | **0.100** | 0.127 | 7.8×10⁻¹⁷² |
| DeepNPTS | **0.100** | 0.424 | < 10⁻³⁰⁰ |
| SeasonalNPTS | **0.100** | 0.226 | < 10⁻³⁰⁰ |
| NPTS | **0.100** | 0.150 | 1.9×10⁻²¹³ |

### Where the intervals actually fail

Averages hide tail behaviour, so this is the number to look at:

| method | datasets with coverage < 0.80 | < 0.50 | worst coverage |
|---|--:|--:|--:|
| **CSP v0.2.0** | **18** | **1** | **0.487** |
| Nixtla CSP | 42 | 14 | 0.266 |
| NPTS | 50 | 16 | 0.166 |
| SeasonalNPTS | 99 | 38 | 0.060 |
| DeepNPTS | 160 | 89 | **0.000** |

DeepNPTS produces a nominal-95% interval that contains the truth at **no horizon at all** in
roughly a fifth of windows. CSP v0.2.0 has one dataset below 0.50 and none below 0.30.

### Cost

Both timed on the same machine (MacBook Air, single-threaded) over the 218-dataset sweep:

| | per forecast record | total CPU | models trained |
|---|--:|--:|--:|
| **CSP v0.2.0** | **5.99 ms** | **~2.5 min** | **0** |
| DeepNPTS | 5,300 ms | 35.4 h | 1,165 |

**DeepNPTS is 885× slower.**

CSP trains nothing — no weights, no epochs, no hyper-parameter search. DeepNPTS's 35.4 h is the
sum across four parallel shards (8.6 h wall clock).

Even granting DeepNPTS its best case — training once per dataset rather than once per
rolling-origin window, which its own protocol does not do — it needs 1,005 ms/record and is
still **168× slower**.

### Scope

The lead is largest on the interval score and on calibration, and narrower on CRPS — CRPS averages
over the whole predictive distribution and is forgiving of intervals that are sharp in the wrong
place, while the Winkler score charges 2/α for every miss. Both are reported above. v0.2.0 trades
roughly 2% of scaled CRPS for the calibration gains described in `CHANGELOG.md`.

Reproduction scripts and the full per-window result files accompany the CSP paper.

## Install

Not on PyPI yet — install from GitHub:

```bash
pip install git+https://github.com/valeman/csp-forecaster.git
# or, from a local clone, for development:
#   pip install -e .
# dependency: numpy only
```

## Use

```python
import numpy as np
from csp_forecaster import ConformalSeasonalPool

history = np.asarray(my_series)                      # 1-D array
csp = ConformalSeasonalPool(adaptive=True, mode="fast", random_state=0)
csp.fit(history, seasonal_period=24)
result = csp.predict(H=24, alpha=0.05, n_samples=100)

result.samples      # (H, n_samples) predictive draws
result.lower, result.upper   # (H,) central 1-alpha interval
result.quantiles    # {tau: (H,)} for the M5 quantile levels
```

`adaptive=True` → "CSP-Adaptive"; `adaptive=False` → "CSP-Fixed".

## Two execution paths

| `mode` | RNG | Speed | Use when |
|---|---|---|---|
| `"legacy"` | global NumPy RNG | reference | Reproducing the published per-window numbers **bit-exactly** |
| `"fast"` | explicit seeded `Generator` | ~1.2× faster, vectorized quantiles, float32 samples, reproducible | Everything else |

The paths build the **same** residual/pool construction and target the same
predictive distribution. They are **not** bit-identical: vectorized draws differ
from the per-horizon loop, and CSP's distribution is bimodal (pool vs conformal),
so central quantiles are ill-conditioned in the gap between modes. They agree on
the quantities the paper reports — **CRPS and coverage** — which is the correct
equivalence criterion.

One deliberate difference: with **no seasonality (`m=1`)** the seasonal pool is
undefined. The legacy path reproduces the original code (draws a half-budget
conformal sample and tiles it up to `n_samples`); the fast path does the cleaner
thing and draws the full budget from the conformal component.

## Options — what each one does, and when to use it

`ConformalSeasonalPool` exposes five behavioural knobs.

> **v0.2.0 fixes a correctness bug on short series.** With `residual_mode="h_step"` (the
> default) the per-horizon residual pool was drawn from the calibration window only, so it
> *shrank* as the horizon grew and could starve to a single element — producing a 95%
> interval of **width exactly 0.0** and **coverage exactly 0.0**. The pool now widens its
> search before compromising the lag. Long-series results are unchanged (verified
> bit-identical); short-series calibration improves substantially, e.g. `m3_yearly` mean
> coverage 0.300 → 0.650. See `CHANGELOG.md`.

> **Defaults changed in v0.1.4.** The defaults are now the **recommended, best-scoring
> configuration** — `residual_mode="h_step"`, `decay_unit="step"`, `orientation=False` —
> which wins on CRPS *and* the Winkler/interval score *and* sharpness across a 20-dataset
> benchmark, while fixing non-seasonal coverage. To reproduce the **original paper** behaviour
> exactly, set `mode="legacy", residual_mode="paper", decay_unit="cycle", orientation=False`.

### `adaptive` (bool, default `True`)

Chooses the variant. `adaptive=True` (**CSP-Adaptive**) turns the seasonal pool *off*
when there is no seasonality (`m≤1`) and down-weights it when fewer than three full
seasonal cycles are available; `adaptive=False` (**CSP-Fixed**) always mixes the pool at
`pool_weight`. Use Adaptive as the general default; Fixed only if you specifically want a
constant pool weight regardless of history depth.

### `mode` ({"fast", "legacy"}, default `"fast"`)

Implementation path, *not* a modelling choice. `"fast"` is vectorized with a seeded
generator (reproducible, float32 samples); `"legacy"` is the original per-horizon loop on
the global RNG and is **bit-exact** with the published code. Use `"legacy"` only to
reproduce paper numbers exactly; `"fast"` otherwise.

### `residual_mode` ({"paper", "h_step"}, default `"h_step"`)

How the conformal residual pool is built **across the horizon**.

- `"paper"` — one residual pool (seasonal lag `m`, or 1-step differences when `m=1`) reused
  for every horizon. Interval width is then *constant* across horizons. For seasonal data
  with `H≤m` this is exactly right, but for non-seasonal (`m=1`) or long-horizon (`H>m`)
  series the far-horizon intervals are too narrow and **coverage decays with horizon**.
- `"h_step"` — the pool is indexed by horizon with the seasonal-naive multi-step lag
  `L_h = m·⌈h/m⌉`. For `h≤m` this equals `m` (so seasonal short-horizon forecasts are
  **unchanged**); for `m=1` it equals `h`, so the interval **widens with horizon** and
  coverage stays near nominal.

*Why / when:* keep `"paper"` to reproduce the published results, or when all your forecasts
are seasonal with `H≤m`. Switch to `"h_step"` when you forecast **non-seasonal series or
horizons longer than one season** — on `exchange_rate` (m=1, H=30) it lifts coverage from
0.49 → 0.94 *and* improves CRPS, at no cost to the seasonal datasets.

### `decay_unit` ({"cycle", "step"}, default `"step"`)

Unit for the seasonal-pool exponential recency decay (rate `exp_lambda`).

- `"step"` — decay by **absolute observation age** (time steps). Same-phase observations one
  season apart are `m` steps apart, so this weights recent cycles far more heavily and
  concentrates the pool on the recent regime. **Best CRPS and Winkler** in benchmarking.
- `"cycle"` — decay by **cycle age** (the original paper behaviour); with the same `exp_lambda`
  it is `m`× weaker than `"step"`.

*Why / when:* `"step"` is the single biggest driver of CRPS/sharpness quality and is the new
default. Use `"cycle"` only to reproduce the published paper numbers. (Note: `exp_lambda` means
different things under the two units — `0.01` per *step* ≈ `0.01·m` per *cycle*.)

### `orientation` (bool, default `False`)

A finite-sample (conformal) correction to the interval quantiles: a lower quantile `q` is
read at `⌊(n+1)q⌋/n` and an upper at `⌈(n+1)q⌉/n` instead of plain `q`, pushing the bounds
slightly outward (`n` = number of samples).

- `orientation=False` (default) — plain empirical quantiles: the **sharpest** intervals, and
  the best **CRPS** *and* **Winkler/interval score**.
- `orientation=True` — higher raw **coverage** (closer to nominal) but **wider** intervals.

*Why / when:* it only changes the reported quantiles, **never the samples — so CRPS is
unaffected**. It does, however, *worsen* the Winkler/interval score, because the extra width is
penalised. Turn it on only when hitting the nominal coverage level is the priority and the wider
intervals are acceptable.

### Choosing a configuration

The **defaults** (`residual_mode="h_step", decay_unit="step", orientation=False`) are the
recommended general config — best CRPS, best Winkler, sharpest intervals, calibrated multi-step.

| Goal | Configuration |
|---|---|
| **General use (recommended)** | **defaults** |
| Reproduce the paper exactly | `mode="legacy", residual_mode="paper", decay_unit="cycle", orientation=False` |
| Maximise nominal coverage (accept wider intervals) | defaults + `orientation=True` |

```python
# Recommended (this is the default):
ConformalSeasonalPool().fit(y, m).predict(H)
# Exact paper reproduction:
ConformalSeasonalPool(mode="legacy", residual_mode="paper",
                      decay_unit="cycle", orientation=False).fit(y, m).predict(H)
```

## Nixtla / statsforecast integration

CSP is **not** natively the statsforecast interface (it uses `predict(H, alpha, …)`
and returns a dataclass), but `csp_forecaster.nixtla.CSPModel` adapts it to the
statsforecast per-model protocol so it drops into the orchestrator. The adapter
is pure NumPy; `statsforecast` is only needed if you use `StatsForecast(...)`.

```python
from csp_forecaster import CSPModel

# Standalone, statsforecast-style:
m = CSPModel(season_length=24, alias="CSP", mode="fast").fit(y)   # y: 1-D array
m.predict(h=24, level=[80, 95])      # -> {'mean', 'lo-80','hi-80','lo-95','hi-95'}
m.forecast(y, h=24, level=[95])      # one-shot fit+predict

# Inside the orchestrator (long df: unique_id, ds, y):
from statsforecast import StatsForecast
sf = StatsForecast(models=[CSPModel(season_length=24, alias="CSP")], freq="H")
sf.forecast(df=df, h=24, level=[95])  # columns: CSP, CSP-lo-95, CSP-hi-95
```

`level=L` maps to `alpha = 1 - L/100`; `lo-L`/`hi-L` are the empirical quantiles of
the CSP samples; `mean` is the per-horizon sample mean.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

The suite (NumPy only) covers:

- **Fast vs legacy equivalence** — the two modes agree on CRPS and empirical coverage to within Monte-Carlo tolerance.
- **Legacy reproducibility** — a fixed global seed yields identical samples.
- **statsforecast adapter** — `CSPModel` returns the expected `mean` / `lo-L` / `hi-L` outputs and runs inside `StatsForecast` (when it is installed).

## Citation

CSP is introduced in *Training-Free Probabilistic Time-Series Forecasting with Conformal
Seasonal Pools* (V. Manokhin, 2026), **accepted at COPA 2026** and published in the
**Proceedings of Machine Learning Research** (PMLR vol. 329, via JMLR).
Preprint: arXiv:2605.03789 — <https://arxiv.org/abs/2605.03789>.

```bibtex
@inproceedings{manokhin2026csp,
  title     = {Training-Free Probabilistic Time-Series Forecasting with Conformal Seasonal Pools},
  author    = {Manokhin, Valery},
  booktitle = {Proceedings of the Symposium on Conformal and Probabilistic Prediction
               with Applications (COPA)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {329},
  year      = {2026},
  publisher = {PMLR},
  eprint    = {2605.03789},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url       = {https://arxiv.org/abs/2605.03789}
}
```

A practitioner-facing companion article is **forthcoming in *Foresight: The International
Journal of Applied Forecasting***.

A companion paper — *Report the Floor: A Training-Free Conformal Interval Is a
Mandatory Baseline for Probabilistic Time-Series Forecasting* (arXiv:2606.09473) —
benchmarks CSP against the trivial conformal-naive floors.
