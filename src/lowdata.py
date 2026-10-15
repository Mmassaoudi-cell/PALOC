"""Low-data experiment (post-freeze): train on a fraction of TRAIN normals; VAL/TEST unchanged.
All preprocessing statistics are recomputed from the retained training subset only."""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
from common import RES_DIR, Dataset, evaluate_scores, set_seed
from grid import Grid
from registry import REGISTRY

FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def subsample(ds, frac, seed):
    rng = np.random.default_rng(1234 + seed)
    tr = ds.idx["train"]
    keep = np.sort(rng.choice(tr, max(int(round(frac * len(tr))), 16), replace=False))
    ds.idx["train"] = keep
    Z = ds.Z[keep].astype(np.float64)
    ds.mu, ds.sd = Z.mean(0), Z.std(0) + 1e-8
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["ieee14p2", "ieee118p2"])
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fracs", nargs="+", type=float, default=FRACS)
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    extra = json.loads(a.extra)
    out = os.path.join(RES_DIR, "lowdata", "lowdata.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    def read_rows():
        return pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    for s in a.systems:
        g = Grid(s)
        for r in a.reps:
            for f in a.fracs:
                for m in a.methods:
                    label = a.name or m
                    if any(x["method"] == label and x["system"] == s and x["rep"] == r and abs(x["frac"] - f) < 1e-9 for x in read_rows()):
                        continue
                    set_seed(r)
                    ds = subsample(Dataset(s, r), f, r)
                    det = REGISTRY[m](ds, g, seed=r, **extra.get(m, {}))
                    sets = {k: v for k, v in ds.eval_sets().items() if k in ("val_N", "val_A", "test_N", "test_A_S", "test_A_M", "test_A_W")}
                    sc = {k: det.score(v) for k, v in sets.items()}
                    new = [dict(method=label, system=s, rep=r, frac=f, n_train=len(ds.idx["train"]), **x)
                           for x in evaluate_scores(sc, alphas=(0.05,)) if x["set"] in ("S", "M", "W")]
                    rows = read_rows() + new  # merge-on-write: safe for concurrent instances
                    pd.DataFrame(rows).drop_duplicates(subset=["method", "system", "rep", "frac", "set"], keep="last").to_csv(out, index=False)
                    print(label, s, r, f, flush=True)
