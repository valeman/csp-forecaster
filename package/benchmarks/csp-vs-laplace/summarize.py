"""Dataset-level summary tables for the CSP vs laplace comparison (reads per_window_metrics.csv.gz)."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

W = pd.read_csv("per_window_metrics.csv.gz", dtype={"series_id": str})
M = ["crps_scaled", "msis", "winkler", "mase", "rmsse", "coverage", "width_scaled", "bias"]


def pivots(D):
    g = D.groupby(["dataset", "method"])[M].mean().reset_index()
    return {k: g.pivot(index="dataset", columns="method", values=k) for k in M}


def summary(D):
    P = pivots(D); n = len(P["msis"]); out = {}
    for k, lab in [("crps_scaled", "CRPS"), ("msis", "MSIS"), ("winkler", "Winkler"), ("mase", "MASE"), ("rmsse", "RMSSE")]:
        c, l = P[k]["CSP"], P[k]["laplace"]
        r = np.log((c / l).clip(1e-9, 1e9))
        out[lab] = dict(csp_wins=int((c < l).sum()), laplace_wins=int((l < c).sum()), n=n,
                        csp_median=float(c.median()), laplace_median=float(l.median()),
                        ratio=float(np.exp(r.mean())), p=float(wilcoxon(c - l).pvalue))
    return pd.DataFrame(out).T, P


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    S, P = summary(W)
    print("Windows:", len(W) // 2, " datasets:", len(P["msis"]))
    print(S.round(4).to_string())
    for m in ["CSP", "laplace"]:
        c = P["coverage"][m]
        print(f"{m}: mean coverage {c.mean():.3f}; datasets <0.85: {(c < 0.85).sum()}; >=0.99: {(c >= 0.99).sum()}; "
              f"median width {P['width_scaled'][m].median():.2f}; median bias {P['bias'][m].median():+.3f}; "
              f"total time {W[W.method == m].runtime_s.sum():.1f} s")
    W["hb"] = pd.cut(W.H, [0, 12, 24, 48, 96, 1000], labels=["1-12", "13-24", "25-48", "49-96", ">96"])
    for hb, D in W.groupby("hb", observed=True):
        s, p = summary(D); t = D.groupby("method").runtime_s.median() * 1000
        print(f"H {hb}: datasets {len(p['msis'])}, CSP wins CRPS {s.loc['CRPS','csp_wins']}, MSIS {s.loc['MSIS','csp_wins']}, "
              f"MASE {s.loc['MASE','csp_wins']}; ms/forecast CSP {t['CSP']:.1f}, laplace {t['laplace']:.0f}")
    W["season"] = np.where(W.m > 1, "seasonal (m>1)", "non-seasonal (m=1)")
    for sg, D in W.groupby("season"):
        s, p = summary(D)
        print(f"{sg}: datasets {len(p['msis'])}, CSP wins CRPS {s.loc['CRPS','csp_wins']}, MSIS {s.loc['MSIS','csp_wins']}, MASE {s.loc['MASE','csp_wins']}")
    r = (P["crps_scaled"]["laplace"] / P["crps_scaled"]["CSP"]).sort_values()
    print("laplace best (CRPS ratio laplace/CSP):", r.head(8).round(2).to_dict())
    print("laplace worst:", r.tail(8).round(1).to_dict())
    print("laplace CRPS more than 10x CSP on", int((r > 10).sum()), "datasets; more than 2x on", int((r > 2).sum()))
    g = W.groupby(["dataset", "method"])[M].mean().round(6).unstack()
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g.insert(0, "windows", W[W.method == "CSP"].groupby("dataset").size()); g.insert(1, "H", W.groupby("dataset").H.first()); g.insert(2, "m", W.groupby("dataset").m.first())
    g.to_csv("per_dataset.csv")
