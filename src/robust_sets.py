"""Deterministic robustness test sets (built from TEST operating points only; cached per replicate).

  rob_noise{k}_N / _A_M / _A_W : measurement noise scaled by k in {0.5, 1.5, 2.0}
  rob_miss{q}_N / _A_M / _A_W  : fraction q in {0.05, 0.10} of channels missing at random,
                                  imputed with the TRAIN mean (identical for every method)
  sweep{s}_A                    : fresh AC-stealthy attacks with magnitude ranges = s x tier-S ranges
                                  s in {0.02, 0.05, 0.1, 0.2, 0.5}; evaluated against test normals
Thresholds always come from nominal validation normals (operators calibrate under nominal conditions).
"""
import os

import numpy as np
import torch

from common import DATA_DIR

NOISE = [0.5, 1.5, 2.0]
MISS = [0.05, 0.10]
SWEEP = [0.02, 0.05, 0.1, 0.2, 0.5]


def _h(g, X):
    return g.h(torch.tensor(X, dtype=torch.float64)).numpy()


def build(ds, g):
    f = os.path.join(DATA_DIR, f"robust_{ds.system}_rep{ds.rep}.npz")
    if os.path.exists(f):
        d = np.load(f)
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(10_000 + 97 * ds.rep + sum(map(ord, ds.system)))
    Xt = ds.X[ds.idx["test"]]
    h0 = _h(g, Xt)
    att = {}
    for t in ("M", "W"):
        xa, base = ds.att(f"test_{t}", "xa"), ds.att(f"test_{t}", "base")
        att[t] = (_h(g, xa) - _h(g, ds.X[base]), _h(g, ds.X[base]))
    out = {}
    for k in NOISE:
        out[f"rob_noise{k}_N"] = h0 + k * rng.normal(size=h0.shape) * g.sigma
        for t, (a, hb) in att.items():
            out[f"rob_noise{k}_A_{t}"] = hb + a + k * rng.normal(size=a.shape) * g.sigma
    for q in MISS:
        Zn = h0 + rng.normal(size=h0.shape) * g.sigma
        sets = {"N": Zn}
        for t, (a, hb) in att.items():
            sets[f"A_{t}"] = hb + a + rng.normal(size=a.shape) * g.sigma
        for k, Z in sets.items():
            M = rng.random(Z.shape) < q
            Z = np.where(M, ds.mu[None, :], Z)
            out[f"rob_miss{q}_{k}"] = Z
    # attack-strength sweep
    from datagen import TIERS
    v0, v1, a0, a1 = TIERS["S"]
    elig = ds.d["eligible"]
    for s in SWEEP:
        b = rng.choice(elig, size=len(Xt))
        c = np.zeros_like(Xt)
        dv = rng.uniform(v0 * s, v1 * s, len(Xt)) * rng.choice([-1, 1], len(Xt))
        dt = np.deg2rad(rng.uniform(a0 * s, a1 * s, len(Xt))) * rng.choice([-1, 1], len(Xt))
        c[np.arange(len(Xt)), b] = dv
        pos = np.array([g.N + int(np.where(g.nonref == bb)[0][0]) for bb in b])
        c[np.arange(len(Xt)), pos] = dt
        a = _h(g, Xt + c) - h0
        out[f"sweep{s}_A"] = h0 + a + rng.normal(size=a.shape) * g.sigma
        out[f"sweep{s}_bus"] = b
    out = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in out.items()}
    tmp = f[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, f)
    return out
