"""Localisation (v2). For each method the per-bus statistic is ranked; top-1/top-3/1-hop accuracy is reported for
VALIDATION attacks (used to choose PALOC's localisation read-out) and TEST attacks (single-bus tiers M/W and the
aligned-attack sets).  PALOC v2 exposes two read-outs: 'neural' (d_b of Eq. 9) and 'anchor' (linear LOO statistic)."""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import pandas as pd
import torch

from common import DEV, RES_DIR, Dataset, bus_groups, set_seed
from grid import Grid
from registry import REGISTRY
import evalsets_v2
import ae_models as A


def ae_bus(det, grid, Z):
    G = bus_groups(grid)
    zn = torch.tensor(det.ds.norm(Z), device=DEV)
    m = det.model
    with torch.no_grad():
        if isinstance(m, A.PIAE):
            u = m.E(zn); r = (zn - m.D(u)) ** 2 + (zn - m.P(u)) ** 2
        elif isinstance(m, A.VAE):
            r = (zn - m.D(m.E(zn)[:, : m.n])) ** 2
        else:
            r = (zn - m.D(m.E(zn))) ** 2
    r = r.cpu().numpy()
    return np.stack([r[:, g].mean(1) for g in G], 1)


def glrt_bus(det, Z):
    R = (det.ds.norm(Z).astype(np.float64) - det.mu) @ det.P
    return np.stack([np.einsum("bi,ij,bj->b", R[:, gi], Q, R[:, gi]) for gi, Q in zip(det.groups, det.inv)], 1)


def readouts(method, det, grid):
    if method.startswith("PALOC2"):
        def neural(Z):
            X, _ = det.est(Z)
            S = torch.tensor(det.rep(X), device=DEV)
            with torch.no_grad():
                return torch.cat([det._d2(S[i:i + 4096])[0] for i in range(0, len(S), 4096)]).cpu().numpy()
        return {"neural": neural, "anchor": det.loc_stats}
    if method == "BayesGLRT":
        return {"glrt": lambda Z: glrt_bus(det, Z)}
    return {"chgroup": lambda Z: ae_bus(det, grid, Z)}


def acc(B, true, Adj):
    rank = np.argsort(-B, 1)
    top1 = rank[:, 0] == true
    return dict(top1=top1.mean(), top3=(rank[:, :3] == true[:, None]).any(1).mean(), hop1=(top1 | Adj[rank[:, 0], true]).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["PALOC2", "BayesGLRT", "PIBAE-t", "SAE-t"])
    ap.add_argument("--systems", nargs="+", default=["ieee14p2", "ieee39p2", "ieee118p2"])
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    out = os.path.join(RES_DIR, "localization", "localization_v2.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    for s in a.systems:
        for r in a.reps:
            ds, g = Dataset(s, r), Grid(s)
            v2 = evalsets_v2.build(ds, g)
            Adj = g.adjacency()
            vt = ds.att("val_mix", "tier")
            sets = {"val_M": (ds.att("val_mix")[vt == "M"], ds.att("val_mix", "bus")[vt == "M"]),
                    "val_W": (ds.att("val_mix")[vt == "W"], ds.att("val_mix", "bus")[vt == "W"]),
                    "M": (ds.att("test_M"), ds.att("test_M", "bus")), "W": (ds.att("test_W"), ds.att("test_W", "bus")),
                    "aligM": (v2["test_A_aligM_Z"], v2["test_A_aligM_bus"]), "aligW": (v2["test_A_aligW_Z"], v2["test_A_aligW_bus"])}
            for m in a.methods:
                if any(x["method"] == m and x["system"] == s and x["rep"] == r for x in rows):
                    continue
                set_seed(r)
                det = REGISTRY[m](ds, g, seed=r)
                for ro, fn in readouts(m, det, g).items():
                    for k, (Z, true) in sets.items():
                        rows.append(dict(method=m, readout=ro, system=s, rep=r, set=k, chance=1 / len(ds.d["eligible"]), **acc(fn(Z), true, Adj)))
                pd.DataFrame(rows).to_csv(out, index=False)
                print(m, s, r, flush=True)
