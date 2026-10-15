"""Validation-only hyper-parameter selection for classical unsupervised benchmarks.
Replicate 0 only; criterion = validation macro-F1 at the conformal 95% threshold (val normals + val_mix attacks).
TEST data are never loaded. Output: configs/benchmark_hp.json and logs/tuning/benchmarks.csv"""
import itertools, json, os, sys
import numpy as np, pandas as pd
from common import Dataset, ROOT, evaluate_scores
from grid import Grid
from registry import REGISTRY

GRID = {
    "PCA": {"k": [2, 5, 10, 15, 20, 30, 50, 80, 120, 200]},
    "IForest": {"n_estimators": [100, 300], "max_features": [0.5, 1.0]},
    "OCSVM": {"nu": [0.01, 0.05, 0.1], "gamma": [0.1, 0.3, 1.0, 3.0]},
    "kNN": {"k": [1, 3, 5, 10, 20]},
}
SYS = sys.argv[1:] or ["ieee14", "ieee39", "ieee118"]
out_json = os.path.join(ROOT, "configs", "benchmark_hp.json")
best = json.load(open(out_json)) if os.path.exists(out_json) else {}
rows = []
for system in SYS:
    ds, g = Dataset(system, 0), Grid(system)
    sets = {k: v for k, v in ds.eval_sets().items() if k.startswith("val")}
    for name, grid in GRID.items():
        for vals in itertools.product(*grid.values()):
            p = dict(zip(grid.keys(), vals))
            if name == "PCA" and p["k"] >= g.m:
                continue
            det = REGISTRY[name](ds, g, seed=0, **p)
            sc = {k: det.score(v) for k, v in sets.items()}
            r = [x for x in evaluate_scores(sc, alphas=(0.05,)) if x["set"] == "val_mix"][0]
            rows.append(dict(system=system, method=name, **p, val_macro_f1=r["macro_f1"], val_auc=r["auc"]))
            print(rows[-1], flush=True)
        df = pd.DataFrame([x for x in rows if x["system"] == system and x["method"] == name])
        b = df.sort_values(["val_macro_f1", "val_auc"], ascending=False).iloc[0]
        best.setdefault(name, {})[system] = {k: (int(b[k]) if k in ("k", "n_estimators") else float(b[k])) for k in grid}
os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True); os.makedirs(os.path.join(ROOT, "logs", "tuning"), exist_ok=True)
json.dump(best, open(os.path.join(ROOT, "configs", "benchmark_hp.json"), "w"), indent=1)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "logs", "tuning", f"benchmarks_{'_'.join(SYS)}.csv"), index=False)
print(json.dumps(best, indent=1))
