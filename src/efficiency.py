"""Efficiency measurement: train each detector once (replicate 0) and measure
  * CPU single-sample latency (batch = 1, 1 thread; the source paper's setting): mean, p95
  * batched throughput (samples/s, batch 1024) on the default device of this process
  * trainable parameters, model size (MB, fp32), MACs per sample (nn.Linear hooks + analytic physics cost)
  * fit (training) time
Run: RUN_DEVICE=cpu OMP_NUM_THREADS=1 python efficiency.py --methods PIBAE ... --systems ieee14 ieee118 --protocol p2
"""
import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
from common import RES_DIR, ROOT, Dataset, set_seed
from grid import Grid
from registry import REGISTRY


def h_macs(g):
    nnz = lambda M: int((np.abs(M) > 0).sum())
    return 4 * nnz(g.Ybus) + 4 * nnz(g.Yf) + 4 * nnz(g.Yt) + 12 * g.N + 16 * g.L  # sparse complex MV as 4 real MACs/nnz


def torch_modules(det):
    mods = []
    for name in ("model",):
        if hasattr(det, name) and isinstance(getattr(det, name), nn.Module):
            mods.append(getattr(det, name))
    if hasattr(det, "est") and hasattr(det.est, "__self__") and isinstance(det.est.__self__, nn.Module):
        mods.append(det.est.__self__)
    return mods


def count_linear_macs(det, Z1):
    tot = [0]
    hooks = []
    for mod in torch_modules(det):
        for m in mod.modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(lambda m, i, o: tot.__setitem__(0, tot[0] + o.numel() // o.shape[-1] * m.in_features * m.out_features // max(1, i[0].shape[0]) if i[0].dim() > 1 else 0)))
    det.score(Z1)
    for h in hooks:
        h.remove()
    return tot[0]


def physics_macs(method, det, g, kw):
    if method in ("PIBAE", "PIAE", "PIBAEHL", "PIBAE-t", "PIAE-t"):
        return h_macs(g)
    if hasattr(det, "se") and det.se is not None and hasattr(det.se, "G0"):
        it = det.se.iters
        return it * (g.n * g.m + h_macs(g)) + h_macs(g)
    if method == "BDD":
        return np.nan  # iterative WLS with Jacobian + dense solve: reported via latency
    return 0


def n_params(det):
    if hasattr(det, "info"):
        p = det.info().get("params", np.nan)
        return p
    return np.nan


def measure(method, system, kw, n_lat=300):
    set_seed(0)
    ds, g = Dataset(system, 0), Grid(system)
    t0 = time.time()
    det = REGISTRY[method](ds, g, seed=0, **kw)
    fit = time.time() - t0
    Z = ds.normals("test")
    det.score(Z[:4])  # warm-up
    lat = []
    for i in range(n_lat):
        t = time.perf_counter(); det.score(Z[i:i + 1]); lat.append(time.perf_counter() - t)
    lat = np.array(lat[20:]) * 1e3
    Zb = np.tile(Z, (1, 1))[:1024]
    t = time.perf_counter(); det.score(Zb); thr = len(Zb) / (time.perf_counter() - t)
    try:
        lm = count_linear_macs(det, Z[:1])
    except Exception:
        lm = np.nan
    pm = physics_macs(method, det, g, kw)
    passes = det.info().get("n_mask_passes", 1) if hasattr(det, "info") else 1
    p = n_params(det)
    return dict(method=method, system=system, fit_time_s=fit, lat_mean_ms=lat.mean(), lat_p95_ms=np.percentile(lat, 95),
                throughput_sps=thr, params=p, size_mb=(p * 4 / 2 ** 20 if np.isfinite(p) else np.nan),
                macs_nn=lm, macs_physics=pm, macs_total=(lm if np.isfinite(lm) else 0) + (pm if np.isfinite(pm) else 0),
                mask_passes=passes, device=os.environ.get("RUN_DEVICE", "cuda"), threads=torch.get_num_threads())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["ieee14", "ieee39", "ieee118"])
    ap.add_argument("--extra", default="{}", help='json {method: kwargs}')
    ap.add_argument("--out", default="efficiency.csv")
    a = ap.parse_args()
    extra = json.loads(a.extra)
    out = os.path.join(RES_DIR, "efficiency", a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {(r["method"], r["system"]) for r in rows}
    for s in a.systems:
        for m in a.methods:
            if (m, s) in done:
                continue
            r = measure(m, s, extra.get(m, {}))
            print(r, flush=True)
            rows.append(r)
            pd.DataFrame(rows).to_csv(out, index=False)
