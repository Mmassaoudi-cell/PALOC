"""Run detectors on (system, replicate) datasets; save raw scores and metric rows.

python run.py --methods PIBAE SAE --systems ieee14 --reps 0 1 2 3 4 --tag main
Scores: results/scores/<tag>/<method>_<system>_rep<r>.npz ; metrics: results/raw/<tag>/<method>_<system>_rep<r>.csv
"""
import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "3")
import time
import traceback

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
from common import RES_DIR, Dataset, evaluate_scores, set_seed
from grid import Grid
from registry import REGISTRY


def run_one(method, system, rep, tag, overwrite=False, extra=None, valonly=False):
    out_csv = os.path.join(RES_DIR, "raw", tag, f"{method}_{system}_rep{rep}.csv")
    out_npz = os.path.join(RES_DIR, "scores", tag, f"{method}_{system}_rep{rep}.npz")
    if os.path.exists(out_csv) and not overwrite:
        return pd.read_csv(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True); os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    set_seed(rep)
    from common import DATA_DIR
    f = os.path.join(DATA_DIR, f"{system}_rep{rep}.npz")
    for _ in range(4 * 3600 // 30):  # wait for the data generator (up to 4 h)
        if os.path.exists(f) and time.time() - os.path.getmtime(f) > 20:
            break
        time.sleep(30)
    ds = Dataset(system, rep)
    grid = Grid(system)
    t0 = time.time()
    det = REGISTRY[method](ds, grid, seed=rep, **(extra or {}))
    fit_time = time.time() - t0
    sets = ds.eval_sets()
    if valonly:  # candidate screening: TEST data are never scored
        sets = {k: v for k, v in sets.items() if k.startswith("val") or k == "train_N"}
    t1 = time.time()
    sc = {k: det.score(v) for k, v in sets.items()}
    score_time = time.time() - t1
    n_scored = sum(len(v) for v in sets.values())
    rows = evaluate_scores(sc) if not valonly else evaluate_scores({**sc}, alphas=(0.05, 0.01))
    info = det.info() if hasattr(det, "info") else {}
    for r in rows:
        r.update(method=method, system=system, rep=rep, fit_time=fit_time,
                 score_us_per_sample=1e6 * score_time / n_scored, **{k: v for k, v in info.items() if np.isscalar(v)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    np.savez_compressed(out_npz, **sc)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["ieee14", "ieee39", "ieee118"])
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--valonly", action="store_true")
    ap.add_argument("--extra", default=None, help="json dict of kwargs passed to the method builder")
    a = ap.parse_args()
    extra = json.loads(a.extra) if a.extra else None
    for s in a.systems:
        for r in a.reps:
            for m in a.methods:
                try:
                    df = run_one(m, s, r, a.tag, a.overwrite, extra, a.valonly)
                    v = df[(df.set == "val_mix") & (df.alpha == 0.05)].iloc[0]
                    if a.valonly:
                        print(f"[{a.tag}] {m:14s} {s} rep{r}  VAL: auc={v.auc:.4f} prauc={v.prauc:.4f} mF1={v.macro_f1:.4f} tpr={v.tpr:.3f} fpr={v.fpr:.3f} fit={df.fit_time.iloc[0]:.0f}s", flush=True)
                        continue
                    t = df[df.alpha == 0.05].set_index("set")
                    print(f"[{a.tag}] {m:14s} {s} rep{r}  val: auc={v.auc:.4f} mF1={v.macro_f1:.4f} | test AUC S/M/W="
                          f"{t.loc['S','auc']:.4f}/{t.loc['M','auc']:.4f}/{t.loc['W','auc']:.4f} "
                          f"FPR={t.loc['S','fpr']:.3f} fit={df.fit_time.iloc[0]:.0f}s", flush=True)
                except Exception:
                    traceback.print_exc()
                torch.cuda.empty_cache()
