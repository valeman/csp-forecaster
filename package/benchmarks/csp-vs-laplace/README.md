# CSP vs. laplace (skaters)

**Opponent.** `laplace`, the online ensemble forecaster of the
[`skaters`](https://github.com/microprediction/skaters) package (Cotton), version 0.16.0, called as
`laplace(k=H)` with default settings.

**Why this comparison exists.** The `skaters` README states that "Laplace beats (almost)
everything" and that it gives ground to CSP only "narrowly, on the hard, repeating waveforms". Its
own CSP comparison (`benchmarks/comparisons/laplace-vs-csp`) reports laplace beating CSP on 82 to
99% of FRED series. This folder tests whether that carries over to a large, multi-step benchmark.
It does not: on the CSP benchmark, **CSP beats laplace on 60% of datasets by CRPS and on 64% by
interval score, and is about 7,700 times faster. laplace also slows sharply as the horizon grows,
to the point where the longest horizons could not be run.**

## What was tested there, and what is tested here

| | `skaters` comparison | This comparison |
|---|---|---|
| Forecast horizon | One step ahead only | Multi-step: the horizons of the CSP paper, 1 to 168 steps |
| Data | 4 collections, 1,409 series: FRED daily, weekly and monthly economic change series, and M4-Hourly | 228 datasets from the CSP benchmark: energy, transport, retail, web traffic, finance, economics, health, weather, cloud workloads and the M-competitions |
| Who chose the data | The laplace author | The CSP paper's corpus, fixed before laplace was run |
| Probabilistic metrics | CRPS and log-likelihood | CRPS, interval score (MSIS and Winkler), coverage, interval width |
| Interval width and interval score | Not reported | Reported: laplace's intervals are 44% wider |
| Coverage | Not reported for CSP | Reported for both |
| Point accuracy | Not reported | MASE, RMSSE, bias |
| Runtime | Not reported for this comparison | Reported: laplace is 2,500 to 23,800 times slower, growing with the horizon |
| CSP version | 0.1.4 | 0.2.0 |
| History given to each method | Each method's own | Identical for both |
| laplace's result against CSP | Wins 82 to 99% of FRED series by CRPS; loses M4-Hourly (wins 21%) | Loses: CSP wins 60% of datasets by CRPS, 64% by interval score |

The two results are not in conflict as measurements. The `skaters` comparison is a narrow test:
one step ahead, on economic series from a single source plus one hourly collection, on two metrics.
Its headline claim is much broader than that test. On a broad multi-step benchmark the claim does
not hold. The one-step FRED protocol was not re-run here.

## Protocol

- **Windows.** 2,534 rolling-origin forecast windows from 228 datasets (242 tier-dataset caches) of
  the CSP paper's corpus, listed in `selected_windows.csv`. laplace's cost per observation grows
  with the horizon, so the windows are a budgeted subset: up to six series per dataset and up to
  five origins per series for H ≤ 48, one window per dataset for 48 < H ≤ 168. The single dataset
  with H = 336 is excluded.
- **History.** For each series laplace is warmed up on the last L observations before the first
  selected origin (L = clip(5·m, 400, 720) for H ≤ 48, otherwise 400) and then run forward through
  the later origins. CSP is fitted at each origin on exactly the same span.
- **Seasonal period.** CSP is given the period m. laplace is not; it has built-in seasonal
  candidates for periods 7, 12 and 24.
- **Forecasts.** 95% prediction intervals and 200 samples per step. laplace's interval is read from
  its quantile function; its samples are drawn from that quantile function on a fine grid.
- **Scoring.** CRPS and the Winkler interval score, each divided by the in-sample seasonal-naive
  absolute error (the scaled interval score is MSIS). Point forecasts: sample median for MASE,
  sample mean for RMSSE. Scores are averaged within a dataset; a method wins a dataset when its
  average is lower. Tests are paired Wilcoxon signed-rank tests over datasets.

**Versions.** `csp-forecaster` 0.2.0 (defaults), `skaters` 0.16.0 (`laplace` defaults). Run on
13 November 2026, one core per process, on a shared two-core cloud machine.

## Results

Dataset level, 228 datasets. Lower is better except coverage.

| Metric | CSP wins | laplace wins | CSP median | laplace median | CSP / laplace, typical dataset | p |
|---|---|---|---|---|---|---|
| CRPS | **137 (60%)** | 91 (40%) | **0.824** | 0.960 | 0.65 | < 0.0001 |
| Interval score, MSIS | **147 (64%)** | 81 (36%) | **10.12** | 11.68 | 0.68 | 0.0001 |
| Interval score, Winkler | **147 (64%)** | 81 (36%) | | | 0.68 | 0.0001 |
| MASE | **128 (56%)** | 100 (44%) | **1.128** | 1.213 | 0.64 | 0.0004 |
| RMSSE | **120 (53%)** | 108 (47%) | **0.960** | 1.053 | 0.70 | 0.0055 |

"Typical dataset" is the geometric mean of per-dataset score ratios.

| | CSP | laplace |
|---|---|---|
| Mean coverage (target 0.95) | 0.898 | 0.956 |
| Datasets with coverage below 0.85 | 39 | 13 |
| Datasets with coverage of 0.99 or more | 22 | 76 |
| Median interval width (scaled) | **6.54** | 9.40 |
| Median bias of the point forecast (MASE units) | **-0.02** | -0.24 |
| Total time, 2,534 forecasts | **1.9 s** | 14,664 s |

By forecast horizon (datasets won by CSP, and median time per forecast):

| Horizon | Datasets | CSP wins, CRPS | CSP wins, MSIS | CSP wins, MASE | CSP, ms | laplace, ms | laplace / CSP |
|---|---|---|---|---|---|---|---|
| 1 to 12 | 60 | 31 | 29 | 28 | 0.3 | 750 | 2,500× |
| 13 to 24 | 36 | 28 | 26 | 27 | 0.9 | 5,030 | 5,600× |
| 25 to 48 | 95 | 54 | 65 | 51 | 1.2 | 11,755 | 9,800× |
| 49 to 96 | 30 | 22 | 20 | 20 | 2.4 | 47,304 | 19,700× |
| over 96 | 9 | 3 | 7 | 2 | 4.5 | 107,024 | 23,800× |

By seasonality: on the 184 datasets with a seasonal period CSP wins 112 by CRPS and 125 by MSIS;
on the 45 without one it wins 26 and 23. (One dataset appears with and without a period.)

## Reading

- **On a broad multi-step benchmark CSP is the more accurate method** on about six datasets in ten
  by CRPS and interval score, and on a majority by point accuracy.
- **laplace's intervals are 44% wider.** Its mean coverage is closer to the target (0.956 against
  0.898), and it reaches that by over-covering: 76 of 228 datasets have coverage of 0.99 or more.
  CSP under-covers: 39 datasets are below 0.85. The interval score prices both errors and favours
  CSP on 64% of datasets.
- **At short horizons the two are level** (31 of 60 datasets by CRPS at H ≤ 12). CSP's advantage
  is at horizons of 13 to 96 steps. This is consistent with laplace's own one-step results and
  shows why a one-step test does not describe multi-step behaviour.
- **laplace's point forecast is biased low** (median -0.24 of the naive scale) and is on average
  less accurate than CSP's.
- **Where laplace wins clearly:** Jena weather (CRPS a tenth of CSP's), FRED-MD, rideshare,
  M3 quarterly. Economic and slowly varying series, as its own benchmark shows.
- **Where laplace breaks:** its CRPS is more than ten times CSP's on 9 datasets and more than twice
  on 37. On bitcoin, CIF 2016 and the global-economy panel its predictive distribution is
  numerically unstable and the errors are many orders of magnitude too large.
- **Cost: laplace is thousands of times slower, and the gap widens with the horizon.** In total it
  took 14,664 seconds against 1.9 for CSP, about 7,700 times longer. Per forecast it is 2,500
  times slower than CSP at horizons up to 12 and 23,800 times slower beyond 96: 0.75 seconds per
  forecast at H ≤ 12, 12 seconds at H 25 to 48, 47 seconds at H 49 to 96 and 107 seconds beyond
  that, each with a warm-up of only 400 to 720 observations.
- **At long horizons laplace becomes impractical to run.** Its cost per observation grows roughly
  with the square of the horizon. This is why only 2,534 of the benchmark's 24,502 windows could be
  run, why histories had to be truncated, why horizons above 48 got one window per dataset, and why
  H = 336 was left out altogether: by our cost fit a single H = 336 forecast would take about six
  minutes even with a 400-point warm-up. Running laplace on the full benchmark with full histories
  was out of reach on this machine; CSP runs all 24,502 windows in about 40 seconds.

## A note on `seasonal_anchor`

laplace's `seasonal_anchor` component was added in July 2026 after laplace lost M4-Hourly to CSP's
defaults on 91% of series by CRPS (per the `skaters` benchmark notes). It is an equal blend of a
recency-weighted average of same-phase values and the value one season ago, which is the structure
of CSP's default sampler: a recency-weighted same-phase pool mixed in equal parts with a component
centred on the seasonal-naive forecast.

## Caveats

- laplace ran at default settings, on a subset of the corpus, with a truncated warm-up, and without
  the seasonal period. A longer warm-up or tuned settings may do better.
- One run and one seed.
- The comparison in the `skaters` repository (one step ahead, FRED and M4-Hourly) was not re-run.
- Timings were taken on a shared machine and are indicative.

## Files

| File | Content |
|---|---|
| `per_window_metrics.csv.gz` | One row per window and method: all metrics, history length used, runtime |
| `per_dataset.csv` | Dataset-level means for both methods |
| `selected_windows.csv` | The 2,534 windows (tier, dataset, series, window) |
| `run_comparison.py` | Runs both methods on the selected windows |
| `summarize.py` | Produces every table above from `per_window_metrics.csv.gz` |

## Reproduce

```bash
pip install "git+https://github.com/valeman/csp-forecaster.git#subdirectory=package" skaters==0.16.0 numpy pandas scipy
# windows_<tier>/<dataset>.npz are the rolling-origin window caches of the CSP paper
python run_comparison.py --windows-root /path/to/windows --selection selected_windows.csv --out per_window_metrics.csv
python summarize.py
```

`run_comparison.py` was checked against the stored results: on re-run it reproduces the stored
per-window scores to within 1e-10.
