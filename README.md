# csp_forecaster

A standalone, installable package for **Conformal Seasonal Pools (CSP)** — the
training-free probabilistic time-series forecaster from the paper *Training-Free
Probabilistic Time-Series Forecasting with Conformal Seasonal Pools*.

CSP produces a full predictive sample per horizon by blending two components:

1. a **seasonal pool** — same-phase historical values weighted by exponential recency, and
2. a **conformal residual** component — a seasonal-naive point forecast plus signed calibration residuals.

No training, no neural network, no learned parameters.

## Install

```bash
pip install -e csp_forecaster        # from the repo root
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

## Verification (all runnable with NumPy only)

```bash
# 1. Unit tests: legacy == original (bit-exact), fast ≈ legacy (CRPS/coverage), reproducibility
PYTHONPATH=src:csp_forecaster/src python -m pytest csp_forecaster/tests -q

# 2. Retest on the paper's six datasets vs the audited per-window results
PYTHONPATH=src:csp_forecaster/src python scripts/verify_csp_on_paper_datasets.py --mode legacy
PYTHONPATH=src:csp_forecaster/src python scripts/verify_csp_on_paper_datasets.py --mode fast
```

### Results on the paper's six datasets (380 windows each)

Run against the cached input windows (`results_local/window_cache/`) and compared
to the CSP rows of `results_paper/audited_per_window_t1.csv`:

- **Legacy:** bit-exact with `src/cp_bench/methods.py::ConformalSeasonalPool` under the same global seed (proven across `adaptive`×`m`×`H` configs).
- **Both modes reproduce the audited results:** per-dataset coverage within 0.04 and mean CRPS within ~1.3% for CSP-Adaptive and CSP-Fixed. Remaining gaps are the expected RNG-sequence difference vs the audited run, not implementation error.
- **Fast vs legacy on real data:** CRPS within 0.01%, coverage within 0.0006.

## Provenance

Extracted verbatim (legacy path) from `src/cp_bench/methods.py::ConformalSeasonalPool`.
See `../RESULTS_PROVENANCE.md` and `../PROJECT_MAP.md` for how this fits the wider project.
