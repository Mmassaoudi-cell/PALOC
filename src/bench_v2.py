"""Additional benchmarks (v2).

* BayesGLRT  : Bayesian state-prior GLRT in the spirit of Kosut et al. (IEEE TSG 2011): Gaussian prior on the
               measurement vector (Ledoit-Wolf covariance of TRAIN normals, which equals H Sigma_x H^T + R plus
               linearisation error); for each bus b the GLRT for an unknown attack supported on the bus's channel
               group G_b is T_b = r_G^T (Lambda_GG)^{-1} r_G, r = Lambda (z - mu); score = max_b T_b.
* <AE>-t     : the six deep unsupervised baselines with hyper-parameters tuned by the same Optuna budget and
               validation objective as PALOC (configs/deep_hp.json).
* SupS-*     : supervised references on physics-state features (chord-WLS state + whitened linear LOO residuals).
"""
import json
import os

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

import ae_models as A
from common import DEV, ROOT, bus_groups
from physics import ChordSE
from registry import AEDet, register, n_params


class BayesGLRT:
    def __init__(self, ds, grid, seed=0, **_):
        self.ds = ds
        Z = ds.norm(ds.normals("train")).astype(np.float64)
        self.mu = Z.mean(0)
        self.P = LedoitWolf().fit(Z - self.mu).precision_
        self.groups = bus_groups(grid)
        self.inv = [np.linalg.inv(self.P[np.ix_(gi, gi)]) for gi in self.groups]

    def score(self, Z):
        R = (self.ds.norm(Z).astype(np.float64) - self.mu) @ self.P
        T = np.stack([np.einsum("bi,ij,bj->b", R[:, gi], Q, R[:, gi]) for gi, Q in zip(self.groups, self.inv)], 1)
        return T.max(1)

    def info(self):
        return dict(params=0)


register("BayesGLRT")(BayesGLRT)

# ------------------------------------------------------------------ tuned deep baselines
DEEP_HP = os.path.join(ROOT, "configs", "deep_hp.json")


def deep_hp(name):
    if os.path.exists(DEEP_HP):
        return json.load(open(DEEP_HP)).get(name, {})
    return {}


def build_deep(name, ds, grid, p):
    w = [max(8, int(round(x * p.get("width", 1.0)))) for x in A.WIDTHS[ds.base]]
    lat = max(2, int(round(grid.n * p.get("latent", 1.0))))
    if name == "SAE":
        return A.SAE(grid.m, lat, w)
    if name == "VAE":
        return A.VAE(grid.m, lat, w)
    if name == "GraphAE":
        return A.GraphAE(grid, grid.m, hidden=max(8, int(64 * p.get("width", 1.0))), lat=int(p.get("gae_lat", 2)))
    if name == "PIAE":
        return A.PIAE(grid.m, grid.n, w, A.PhysLayer(grid, ds.mu, ds.sd), w_phy=p.get("w_phy", 1.0))
    if name == "PIBAE":
        return A.PIBAE(grid.m, grid.n, w, A.PhysLayer(grid, ds.mu, ds.sd))
    if name == "DeepSVDD":
        return A.DeepSVDD(grid.m, w, rep=int(p.get("rep", 32)))
    raise KeyError(name)


def _deep(name):
    def b(ds, grid, seed=0, **kw):
        p = {**deep_hp(name), **kw}
        torch.manual_seed(seed)
        model = build_deep(name, ds, grid, p)
        return AEDet(ds, model, lr=p.get("lr", 1e-3), wd=p.get("wd", 1e-5), bs=int(p.get("bs", 128)), epochs=200, patience=20)
    return b


for _n in ("SAE", "VAE", "GraphAE", "PIAE", "PIBAE", "DeepSVDD"):
    register(f"{_n}-t")(_deep(_n))


# ------------------------------------------------------------------ supervised on physics-state features
class StateFeatures:
    """Chord-WLS state (standardised) + whitened linear leave-one-out residual per bus; fitted on TRAIN normals."""

    def __init__(self, ds, grid):
        self.se = ChordSE(grid, iters=3)
        Xtr, _ = self.se.estimate(ds.normals("train"))
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        F = (Xtr - self.mu) / self.sd
        self.P = LedoitWolf().fit(F).precision_
        N = grid.N
        self.idx = [np.array([b] + ([N + int(np.where(grid.nonref == b)[0][0])] if b != grid.ref else [])) for b in range(N)]
        self.Bi = [np.linalg.inv(self.P[np.ix_(i, i)]) for i in self.idx]
        self.Lw = [np.linalg.cholesky(self.P[np.ix_(i, i)]) for i in self.idx]

    def __call__(self, Z):
        X, J = self.se.estimate(Z)
        F = (X - self.mu) / self.sd
        R = F @ self.P
        W = [R[:, i] @ Bi @ L for i, Bi, L in zip(self.idx, self.Bi, self.Lw)]  # whitened LOO residuals
        return np.concatenate([F] + W + [np.log(J[:, None] + 1)], 1).astype(np.float32)


class SupState:
    def __init__(self, ds, grid, fit_fn):
        self.feat = StateFeatures(ds, grid)

        def xy(split):
            key = "train_mix" if split == "train" else "val_mix"
            Zn, Za = ds.normals(split), ds.att(key)
            return self.feat(np.r_[Zn, Za]), np.r_[np.zeros(len(Zn)), np.ones(len(Za))]
        Xtr, ytr = xy("train"); Xv, yv = xy("val")
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        nz = lambda X: (X - self.mu) / self.sd
        self.est, self.extra = fit_fn(nz(Xtr), ytr, nz(Xv), yv)
        self.nz = nz

    def score(self, Z):
        X = self.nz(self.feat(Z))
        if hasattr(self.est, "predict_proba"):
            p = self.est.predict_proba(X)[:, 1]
            return np.log(p + 1e-12) - np.log(1 - p + 1e-12)
        return self.est(X)

    def info(self):
        return self.extra or {}


def _sups(name, fit):
    @register(name)
    def b(ds, grid, seed=0, **kw):
        return SupState(ds, grid, lambda *a: fit(*a, seed=seed))
    return b


def _bootstrap_sup():
    import supervised as S
    _sups("SupS-XGB", S._xgb)
    _sups("SupS-LGBM", S._lgbm)
    _sups("SupS-MLP", lambda X, y, Xv, yv, seed=0: S._torch_fit(S.MLPc(X.shape[1]), X, y, Xv, yv, seed))


_bootstrap_sup()
