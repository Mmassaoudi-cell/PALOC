"""v2 evaluation runner: nominal + adversarial + operating-condition sets, three alpha levels, prevalence-weighted
PR-AUC, nominal and recalibrated thresholds.

python run_v2.py --methods PALOC2 BayesGLRT --systems ieee14p2 --reps 0 1 --tag v2
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
from sklearn.metrics import average_precision_score

from common import DATA_DIR, RES_DIR, Dataset, metrics, set_seed, threshold
from grid import Grid
from registry import REGISTRY
import evalsets_v2

ALPHAS = (0.05, 0.01, 0.001)
N_RECAL = 200


def prauc_prev(s_norm, s_att, prev=0.01):
    y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_att))]
    w = np.r_[np.ones(len(s_norm)), np.full(len(s_att), len(s_norm) * prev / (1 - prev) / max(len(s_att), 1))]
    return average_precision_score(y, np.r_[s_norm, s_att], sample_weight=w)


def row(set_name, alpha, calib, sN, sA, tau):
    r = dict(set=set_name, alpha=alpha, calib=calib, **metrics(sN, sA, tau))
    r["prauc_p1"] = prauc_prev(sN, sA)
    r["n_att"] = len(sA)
    return r


def collect_sets(ds, g):
    S = ds.eval_sets()
    v2 = evalsets_v2.build(ds, g)
    for split in ("val", "test"):
        for k in ("aligM", "aligW", "multiM", "multialigM", "lr"):
            key = f"{split}_A_{k}_Z"
            if key in v2 and len(v2[key]):
                S[f"{split}_A_{k}"] = v2[key]
    if ds.system.endswith("p2"):
        for k in evalsets_v2.NOISE:
            S[f"rob_noise{k}_V"] = v2[f"noise{k}_V"][:N_RECAL]
        for q in evalsets_v2.MISS:
            for part in ("N", "A_M", "A_W", "V"):
                S[f"missnan{q}_{part}"] = v2[f"missnan{q}_{part}"] if part != "V" else v2[f"missnan{q}_V"][:N_RECAL]
        S["topo_N"], S["topo_A_M"], S["topo_V"] = v2["topo_N"], v2["topo_A_M"], v2["topo_V"][:N_RECAL]
    return S


def evaluate(sc):
    rows = []
    for a in ALPHAS:
        tau = threshold(sc["val_N"], a)
        rows.append(row("val_mix", a, "nominal", sc["val_N"], sc["val_A"], tau))
        for k in sc:
            if k.startswith("val_A_"):
                rows.append(row("val_" + k[6:], a, "nominal", sc["val_N"], sc[k], tau))
            elif k.startswith("test_A_"):
                rows.append(row(k[7:], a, "nominal", sc["test_N"], sc[k], tau))
            elif k.startswith("sweep"):
                rows.append(row(k, a, "nominal", sc["test_N"], sc[k], tau))
            elif k.startswith("rob_") and "_A_" in k:
                base = k.split("_A_")[0]
                rows.append(row(k, a, "nominal", sc[base + "_N"], sc[k], tau))
                if base + "_V" in sc:
                    rows.append(row(k, a, "recal", sc[base + "_N"], sc[k], threshold(sc[base + "_V"], a)))
            elif k.startswith(("missnan", "topo")) and "_A_" in k:
                base = k.split("_A_")[0]
                rows.append(row(k, a, "nominal", sc[base + "_N"], sc[k], tau))
                rows.append(row(k, a, "recal", sc[base + "_N"], sc[k], threshold(sc[base + "_V"], a)))
    return rows


def run_one(method, system, rep, tag, overwrite=False, extra=None):
    out_csv = os.path.join(RES_DIR, "raw", tag, f"{method}_{system}_rep{rep}.csv")
    out_npz = os.path.join(RES_DIR, "scores", tag, f"{method}_{system}_rep{rep}.npz")
    if os.path.exists(out_csv) and not overwrite:
        return pd.read_csv(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True); os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    f = os.path.join(DATA_DIR, f"{system}_rep{rep}.npz")
    for _ in range(4 * 3600 // 30):
        if os.path.exists(f) and time.time() - os.path.getmtime(f) > 30:
            break
        time.sleep(30)
    set_seed(rep)
    ds, g = Dataset(system, rep), Grid(system)
    sets = collect_sets(ds, g)
    t0 = time.time()
    det = REGISTRY[method](ds, g, seed=rep, **(extra or {}))
    fit_time = time.time() - t0
    native = getattr(det, "supports_missing", False)
    sc = {}
    t1 = time.time()
    for k, Z in sets.items():
        if not native and np.isnan(Z).any():
            Z = np.where(np.isnan(Z), ds.mu[None, :], Z).astype(np.float32)  # train-mean imputation
        sc[k] = det.score(Z)
    score_time = time.time() - t1
    rows = evaluate(sc)
    info = det.info() if hasattr(det, "info") else {}
    for r in rows:
        r.update(method=method, system=system, rep=rep, fit_time=fit_time, native_missing=native,
                 score_us_per_sample=1e6 * score_time / sum(len(v) for v in sets.values()),
                 **{k: v for k, v in info.items() if np.isscalar(v)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    np.savez_compressed(out_npz, **sc)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["ieee14p2", "ieee39p2", "ieee118p2"])
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--extra", default=None)
    a = ap.parse_args()
    extra = json.loads(a.extra) if a.extra else None
    for s in a.systems:
        for r in a.reps:
            for m in a.methods:
                try:
                    df = run_one(m, s, r, a.tag, a.overwrite, extra)
                    v = df[(df.set == "val_mix") & (df.alpha == 0.05)].iloc[0]
                    print(f"[{a.tag}] {m:16s} {s} rep{r} val mF1={v.macro_f1:.4f} fit={df.fit_time.iloc[0]:.0f}s", flush=True)
                except Exception:
                    traceback.print_exc()
                torch.cuda.empty_cache()
