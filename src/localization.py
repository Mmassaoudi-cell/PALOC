"""Attack localisation (task-specific experiment, TEST split, run only after FINAL_MODEL_CONFIG is frozen).

For each attacked test sample the detector ranks buses by a per-bus anomaly statistic:
  * proposed / linear ablation : bus statistic d_b
  * SAE / PI-BAE               : mean squared standardised channel residual over the bus's channel group
Reported: top-1, top-3 accuracy and 1-hop accuracy (top-1 bus is the attacked bus or one of its neighbours),
per tier, on all attacked samples and on detected attacks only (score > conformal 95% threshold).
"""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import pandas as pd
import torch

from common import RES_DIR, ROOT, Dataset, bus_groups, set_seed, threshold, DEV
from grid import Grid
from registry import REGISTRY
import ae_models as A


def ae_bus_scores(det, grid, Z):
    G = bus_groups(grid)
    zn = torch.tensor(det.ds.norm(Z), device=DEV)
    m = det.model
    with torch.no_grad():
        if isinstance(m, A.PIAE):
            u = m.E(zn); r = (zn - m.D(u)) ** 2 + (zn - m.P(u)) ** 2
        else:
            r = (zn - m.D(m.E(zn))) ** 2
    r = r.cpu().numpy()
    return np.stack([r[:, g].mean(1) for g in G], 1)


def run(method, system, rep, kw):
    set_seed(rep)
    ds, g = Dataset(system, rep), Grid(system)
    det = REGISTRY[method](ds, g, seed=rep, **kw)
    if hasattr(det, "bus_stats"):
        bus = lambda Z: det.bus_stats(Z)[0]
    else:
        bus = lambda Z: ae_bus_scores(det, g, Z)
    tau = threshold(det.score(ds.normals("val")), 0.05)
    Adj = g.adjacency()
    rows = []
    for t in ("S", "M", "W"):
        Z, true = ds.att(f"test_{t}"), ds.att(f"test_{t}", "bus")
        B = bus(Z); s = det.score(Z)
        rank = np.argsort(-B, 1)
        top1 = rank[:, 0] == true
        top3 = (rank[:, :3] == true[:, None]).any(1)
        hop1 = top1 | Adj[rank[:, 0], true]
        det_ = s > tau
        rows.append(dict(method=method, system=system, rep=rep, tier=t, top1=top1.mean(), top3=top3.mean(), hop1=hop1.mean(),
                         top1_detected=top1[det_].mean() if det_.any() else np.nan,
                         top3_detected=top3[det_].mean() if det_.any() else np.nan, detected=det_.mean(),
                         chance_top1=1.0 / len(ds.d["eligible"])))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["ieee14p2", "ieee39p2", "ieee118p2"])
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--name", default=None, help="label used in the output instead of the registry name")
    a = ap.parse_args()
    extra = json.loads(a.extra)
    out = os.path.join(RES_DIR, "localization", "localization.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    for s in a.systems:
        for r in a.reps:
            for m in a.methods:
                label = a.name or m
                if any(x["method"] == label and x["system"] == s and x["rep"] == r for x in rows):
                    continue
                new = run(m, s, r, extra.get(m, {}))
                for x in new:
                    x["method"] = label
                rows += new
                pd.DataFrame(rows).to_csv(out, index=False)
                print(label, s, r, [(x["tier"], round(x["top1"], 3), round(x["top3"], 3)) for x in new], flush=True)
