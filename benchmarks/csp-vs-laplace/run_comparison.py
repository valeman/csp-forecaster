"""CSP vs laplace (skaters) on the CSP benchmark windows.

Reproduces per_window_metrics.csv.gz. Both methods receive exactly the same history: for each
series, laplace is warmed up on the last L observations before the first selected forecast origin
(L = clip(5 * m, 400, 720) for H <= 48, else 400) and then run forward through the later origins;
CSP is fitted on the same span at each origin.

Usage:
    python run_comparison.py --windows-root <dir with windows_<tier>/<dataset>.npz> \
                             --selection selected_windows.csv --out per_window_metrics.csv

Requires: numpy, pandas, csp-forecaster 0.2.0, skaters 0.16.0.
"""
import argparse
import time
import zlib

import numpy as np
import pandas as pd

ALPHA, NSAMP, SEED = 0.05, 200, 0
GRID = np.array([.001, .0025, .005, .01, .025, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65,
                 .7, .75, .8, .85, .9, .95, .975, .99, .995, .9975, .999])


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def fin(a):
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


def crps_sample(S, y):
    out = []
    for h in range(S.shape[0]):
        x = np.sort(S[h]); n = x.size
        i = np.arange(1, n + 1)
        t2 = (2.0 / (n * (n - 1))) * (np.sum((i - 1) * x) - np.sum((n - i) * x))
        out.append(np.mean(np.abs(x - y[h])) - 0.5 * t2)
    return float(np.mean(out))


def score(S, lo, hi, fut, hist, m):
    """Metrics for one window. Scales use the full (untruncated) history, as in the CSP paper."""
    k = int(m) if m and m > 1 else 1
    sn_mae = float(np.mean(np.abs(hist[k:] - hist[:-k]))) if len(hist) > k else 1.0
    sn_mae = sn_mae if sn_mae > 0 else 1.0
    sn_rmse = float(np.sqrt(np.mean((hist[k:] - hist[:-k]) ** 2))) if len(hist) > k else 1.0
    sn_rmse = sn_rmse if sn_rmse > 0 else 1.0
    crps = crps_sample(S, fut)
    wink = float(np.mean((hi - lo) + (2 / ALPHA) * np.maximum(lo - fut, 0) + (2 / ALPHA) * np.maximum(fut - hi, 0)))
    med, mean = np.median(S, axis=1), np.mean(S, axis=1)
    return dict(crps=crps, crps_scaled=crps / sn_mae, winkler=wink, msis=wink / sn_mae,
                coverage=float(np.mean((fut >= lo) & (fut <= hi))), width=float(np.mean(hi - lo)),
                width_scaled=float(np.mean(hi - lo)) / sn_mae,
                mase=float(np.mean(np.abs(med - fut))) / sn_mae,
                rmsse=float(np.sqrt(np.mean((mean - fut) ** 2))) / sn_rmse,
                bias=float(np.mean(med - fut)) / sn_mae)


def run(windows_root, selection, out):
    from csp_forecaster import ConformalSeasonalPool
    from skaters import laplace

    sel = pd.read_csv(selection, dtype={"series_id": str})
    rows = []
    for (tier, ds), g in sel.groupby(["tier", "dataset"], sort=False):
        z = np.load(f"{windows_root}/windows_{tier}/{ds}.npz", allow_pickle=True)
        sid, widx, hist, fut = z["sid"].astype(str), z["widx"], z["hist"], z["fut"]
        H, m = int(z["H"]), int(z["m"])
        seen, occs = {}, np.zeros(len(sid), int)
        for i in range(len(sid)):                      # occurrence counter for repeated (series, window) keys
            key = (sid[i], int(widx[i])); occs[i] = seen.get(key, 0); seen[key] = occs[i] + 1
        lcap = int(np.clip(5 * max(1, m), 400, 720)) if H <= 48 else 400
        for s, gs in g.groupby("series_id", sort=False):
            want = set(zip(gs.window_idx.astype(int), gs.occ.astype(int)))
            ii = [i for i in range(len(sid)) if sid[i] == s and (int(widx[i]), int(occs[i])) in want]
            ii.sort(key=lambda i: len(hist[i]))
            start = max(0, len(hist[ii[0]]) - lcap)
            longest = np.asarray(hist[ii[-1]], float)
            ends = {len(hist[i]): i for i in ii}
            meta = lambda i: dict(tier=tier, dataset=ds, series_id=s, window_idx=int(widx[i]), occ=int(occs[i]), H=H, m=m)

            # laplace: one online pass, forecast read off at each selected origin
            f, st, tfeed, nfed, dists = laplace(k=H), None, 0.0, 0, None
            for pos in range(start, len(longest)):
                v = longest[pos]
                if np.isfinite(v):
                    t0 = time.time(); dists, st = f(float(v), st); tfeed += time.time() - t0; nfed += 1
                if pos + 1 in ends:
                    i = ends[pos + 1]; fu = np.asarray(fut[i], float); Hh = len(fu)
                    t0 = time.time()
                    Q = np.maximum.accumulate(np.array([[d.quantile(float(p)) for p in GRID] for d in dists[:Hh]]), axis=1)
                    lo = np.array([d.quantile(ALPHA / 2) for d in dists[:Hh]])
                    hi = np.array([d.quantile(1 - ALPHA / 2) for d in dists[:Hh]])
                    rng = np.random.default_rng(stable_seed(ds, sid[i], widx[i], "Laplace", SEED))
                    u = np.clip(rng.random((Hh, NSAMP)), GRID[0], GRID[-1])
                    S = np.array([np.interp(u[h], GRID, Q[h]) for h in range(Hh)])
                    rows.append(dict(method="laplace", n_used=nfed, runtime_s=tfeed + time.time() - t0,
                                     **meta(i), **score(S, lo, hi, fu, fin(hist[i]), m)))

            # CSP: fitted on the same span at each origin
            for i in ii:
                h = fin(np.asarray(hist[i], float)[start:]); fu = np.asarray(fut[i], float)
                sd = stable_seed(ds, sid[i], widx[i], "CSP-Repo@trunc", SEED); t0 = time.time()
                r = ConformalSeasonalPool(random_state=sd).fit(h, max(1, m)).predict(H=len(fu), alpha=ALPHA, n_samples=NSAMP)
                rows.append(dict(method="CSP", n_used=len(h), runtime_s=time.time() - t0, **meta(i),
                                 **score(np.asarray(r.samples, float), np.asarray(r.lower, float),
                                         np.asarray(r.upper, float), fu, fin(hist[i]), m)))
        pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-root", required=True)
    ap.add_argument("--selection", default="selected_windows.csv")
    ap.add_argument("--out", default="per_window_metrics.csv")
    a = ap.parse_args()
    run(a.windows_root, a.selection, a.out)
