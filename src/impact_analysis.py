"""Attack impact and attribution (post hoc from saved v2 scores; no retraining).
Impact of an attack = largest error it induces in the operator's view of a line flow (MW, 100-MVA base),
max_l |Pf_l(x+c) - Pf_l(x)|, and the voltage-magnitude bias |dV| (p.u.).
Outputs: results/tables/impact_bins.csv (TPR per impact bin), impact_weighted.csv, bustype_attribution.csv."""
import os

import numpy as np
import pandas as pd
import torch

from common import RES_DIR, Dataset, threshold
from grid import Grid

BINS = [0, 2, 5, 10, 25, np.inf]
LABELS = ["<2", "2-5", "5-10", "10-25", ">25"]
METHODS = ["PALOC2", "PALOC2-noNN", "BayesGLRT", "PCA", "StateMaha", "SAE-t", "PIBAE-t", "VAE-t", "GraphAE-t", "SupS-XGB"]
SYSTEMS = ["ieee14p2", "ieee39p2", "ieee118p2"]


def impacts(ds, g, tier):
    xa, base = ds.att(f"test_{tier}", "xa"), ds.att(f"test_{tier}", "base")
    with torch.no_grad():
        za = g.h(torch.tensor(xa, dtype=torch.float64)).numpy(); z0 = g.h(torch.tensor(ds.X[base], dtype=torch.float64)).numpy()
    f0 = g.N + 2 * (g.N - 1)
    flows = np.abs(za[:, f0:f0 + g.L] - z0[:, f0:f0 + g.L]) * g.baseMVA
    dv = np.abs(xa[:, :g.N] - ds.X[base][:, :g.N]).max(1)
    return flows.max(1), dv, ds.att(f"test_{tier}", "bus")


if __name__ == "__main__":
    rows_b, rows_w, rows_t = [], [], []
    for s in SYSTEMS:
        g = Grid(s)
        bt = g.net._ppc["bus"][:, 1].real.astype(int)
        for r in range(10):
            f = os.path.join(RES_DIR, "scores", "v2", f"PALOC2_{s}_rep{r}.npz")
            if not os.path.exists(f):
                continue
            ds = Dataset(s, r)
            imp = {t: impacts(ds, g, t) for t in ("S", "M", "W")}
            for m in METHODS:
                fs = os.path.join(RES_DIR, "scores", "v2abl" if m.startswith("PALOC2-") else "v2", f"{m}_{s}_rep{r}.npz")
                if not os.path.exists(fs):
                    continue
                sc = np.load(fs)
                tau = threshold(sc["val_N"], 0.05)
                det_all, imp_all = [], []
                for t in ("S", "M", "W"):
                    det = sc[f"test_A_{t}"] > tau
                    mw, dv, bus = imp[t]
                    det_all.append(det); imp_all.append(mw)
                    if m in ("PALOC2", "PALOC2-noNN") and t == "W":
                        for typ, lab in ((1, "load (PQ) bus"), (2, "generator (PV) bus")):
                            sel = bt[bus] == typ
                            if sel.any():
                                rows_t.append(dict(method=m, system=s, rep=r, bus_type=lab, tpr=det[sel].mean(), n=int(sel.sum())))
                det_all, imp_all = np.concatenate(det_all), np.concatenate(imp_all)
                cat = pd.cut(imp_all, BINS, labels=LABELS, right=False)
                for lab in LABELS:
                    sel = np.asarray(cat == lab)
                    if sel.any():
                        rows_b.append(dict(method=m, system=s, rep=r, bin=lab, tpr=det_all[sel].mean(), n=int(sel.sum())))
                rows_w.append(dict(method=m, system=s, rep=r, impact_weighted_detection=(imp_all * det_all).sum() / imp_all.sum(),
                                   tpr_impact_ge5=det_all[imp_all >= 5].mean() if (imp_all >= 5).any() else np.nan))
        print(s, flush=True)
    os.makedirs(os.path.join(RES_DIR, "tables"), exist_ok=True)
    B = pd.DataFrame(rows_b); W = pd.DataFrame(rows_w); T = pd.DataFrame(rows_t)
    B.to_csv(os.path.join(RES_DIR, "tables", "impact_bins.csv"), index=False)
    W.to_csv(os.path.join(RES_DIR, "tables", "impact_weighted.csv"), index=False)
    T.to_csv(os.path.join(RES_DIR, "tables", "bustype_attribution.csv"), index=False)
    print(B.groupby(["method", "bin"]).tpr.mean().unstack().round(3))
    print(W.groupby("method")[["impact_weighted_detection", "tpr_impact_ge5"]].mean().round(4))
    print(T.groupby(["method", "bus_type"]).tpr.mean().unstack().round(4))
