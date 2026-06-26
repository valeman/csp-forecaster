# csp_forecaster

A standalone, installable package for **Conformal Seasonal Pools (CSP)** — the
training-free probabilistic time-series forecaster from the paper *Training-Free
Probabilistic Time-Series Forecasting with Conformal Seasonal Pools*.

CSP produces a full predictive sample per horizon by blending two components:

1. a **seasonal pool** — same-phase historical values weighted by exponential recency, and
2. a **conformal residual** component — a seasonal-naive point forecast plus signed calibration residuals.

No training, no neural network, no learned parameters.

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

CSP is introduced in *Training-Free Probabilistic Time-Series Forecasting with
Conformal Seasonal Pools* (V. Manokhin, 2026), arXiv:2605.03789 —
<https://arxiv.org/abs/2605.03789>. If you use this package, please cite the paper:

```bibtex
@misc{manokhin2026csp,
  title         = {Training-Free Probabilistic Time-Series Forecasting with Conformal Seasonal Pools},
  author        = {Manokhin, Valery},
  year          = {2026},
  eprint        = {2605.03789},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url           = {https://arxiv.org/abs/2605.03789}
}
```

A companion paper — *Report the Floor: A Training-Free Conformal Interval Is a
Mandatory Baseline for Probabilistic Time-Series Forecasting* (arXiv:2606.09473) —
benchmarks CSP against the trivial conformal-naive floors.
