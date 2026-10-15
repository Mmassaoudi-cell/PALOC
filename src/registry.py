"""Method registry: every detector exposes .score(Z_raw) -> anomaly score (higher = more anomalous)."""
import re

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

import ae_models as A
from common import DEV
from physics import ChordSE

REGISTRY = {}


def register(name):
    def deco(f):
        REGISTRY[name] = f
        return f
    return deco


def hp_get(name, ds):
    """Benchmark hyper-parameters: exact system key, else its P2 parent (protocol variants), else the base system."""
    tab = HP[name]
    m = re.match(r"(ieee[0-9]+p2)", ds.system)
    for key in (ds.system, m.group(1) if m else None, ds.base):
        if key in tab:
            return tab[key]
    raise KeyError(name)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AEDet:
    def __init__(self, ds, model, **train_kw):
        self.ds = ds
        Ztr = ds.norm(ds.normals("train")); Zv = ds.norm(ds.normals("val"))
        self.model, self.tinfo = A.train(model, Ztr, Zv, **train_kw)

    def score(self, Z):
        return A.score_np(self.model, self.ds.norm(Z))

    def info(self):
        return dict(params=n_params(self.model), epochs=self.tinfo["epochs"], train_time=self.tinfo["train_time"])


def _phys(ds, grid):
    return A.PhysLayer(grid, ds.mu, ds.sd)


# ------------------------------------------------------------------ source method + its baselines
@register("PIBAE")
def _pibae(ds, grid, seed=0, **kw):
    w = A.WIDTHS[ds.base]
    return AEDet(ds, A.PIBAE(grid.m, grid.n, w, _phys(ds, grid)), **kw)


@register("PIAE")
def _piae(ds, grid, seed=0, w_phy=1.0, **kw):
    w = A.WIDTHS[ds.base]
    return AEDet(ds, A.PIAE(grid.m, grid.n, w, _phys(ds, grid), w_phy=w_phy), **kw)


@register("SAE")
def _sae(ds, grid, seed=0, **kw):
    return AEDet(ds, A.SAE(grid.m, grid.n, A.WIDTHS[ds.base]), **kw)


@register("VAE")
def _vae(ds, grid, seed=0, **kw):
    return AEDet(ds, A.VAE(grid.m, grid.n, A.WIDTHS[ds.base]), **kw)


@register("GraphAE")
def _gae(ds, grid, seed=0, **kw):
    return AEDet(ds, A.GraphAE(grid, grid.m), **kw)


@register("DeepSVDD")
def _dsvdd(ds, grid, seed=0, **kw):
    return AEDet(ds, A.DeepSVDD(grid.m, A.WIDTHS[ds.base]), **kw)


# ------------------------------------------------------------------ classical / physics baselines
class BDD:
    """Chi-square bad-data detection: J(x_hat) of full Gauss-Newton WLS (flat start)."""

    def __init__(self, ds, grid, seed=0):
        self.g = grid

    def score(self, Z):
        return self.g.wls(Z, device=DEV)[1]

    def info(self):
        return dict(params=0)


register("BDD")(BDD)


class SkDet:
    def __init__(self, ds, est, score_fn):
        self.ds, self.est, self.f = ds, est, score_fn
        self.est.fit(ds.norm(ds.normals("train")))

    def score(self, Z):
        return self.f(self.est, self.ds.norm(Z))


@register("PCA")
def _pca(ds, grid, seed=0, k=None, **kw):
    k = k or hp_get("PCA", ds)["k"]
    f = lambda e, X: ((X - e.inverse_transform(e.transform(X))) ** 2).sum(1)
    d = SkDet(ds, PCA(k, random_state=seed), f)
    d.info = lambda: dict(params=k * grid.m)
    return d


@register("IForest")
def _if(ds, grid, seed=0, **kw):
    p = {**hp_get("IForest", ds), **kw}
    return SkDet(ds, IsolationForest(n_estimators=p["n_estimators"], max_features=p["max_features"],
                                     random_state=seed, n_jobs=8), lambda e, X: -e.score_samples(X))


@register("OCSVM")
def _ocsvm(ds, grid, seed=0, **kw):
    p = {**hp_get("OCSVM", ds), **kw}
    return SkDet(ds, OneClassSVM(nu=p["nu"], gamma=p["gamma"] / grid.m), lambda e, X: -e.decision_function(X))


@register("kNN")
def _knn(ds, grid, seed=0, **kw):
    p = {**hp_get("kNN", ds), **kw}

    class K:
        def __init__(s):
            s.nn = NearestNeighbors(n_neighbors=p["k"]).fit(ds.norm(ds.normals("train")))

        def score(s, Z):
            d, _ = s.nn.kneighbors(ds.norm(Z))
            return d.mean(1)
    return K()


class StateMaha:
    """Physics-classical: chord-WLS state estimate + Ledoit-Wolf Mahalanobis distance (train normals)."""

    def __init__(self, ds, grid, seed=0, iters=3):
        self.se = ChordSE(grid, iters=iters)
        Xtr, _ = self.se.estimate(ds.normals("train"))
        self.mu = Xtr.mean(0)
        self.P = LedoitWolf().fit(Xtr - self.mu).precision_

    def score(self, Z):
        X, _ = self.se.estimate(Z)
        D = X - self.mu
        return np.einsum("ij,jk,ik->i", D, self.P, D)


register("StateMaha")(StateMaha)

# validation-tuned hyper-parameters of benchmarks (filled by tune_benchmarks.py; defaults below)
HP = {
    "PCA": {s: {"k": 20} for s in ("ieee14", "ieee39", "ieee118")},
    "IForest": {s: {"n_estimators": 300, "max_features": 1.0} for s in ("ieee14", "ieee39", "ieee118")},
    "OCSVM": {s: {"nu": 0.05, "gamma": 1.0} for s in ("ieee14", "ieee39", "ieee118")},
    "kNN": {s: {"k": 5} for s in ("ieee14", "ieee39", "ieee118")},
}
try:
    import json, os
    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "benchmark_hp.json")
    if os.path.exists(_p):
        for k, v in json.load(open(_p)).items():
            HP[k] = v
except Exception:
    pass

import candidates  # noqa: E402,F401  (registers candidate models)
try:
    import supervised  # noqa: E402,F401  (registers supervised reference models)
except ImportError:
    pass

import paloc2  # noqa: E402,F401  (PALOC v2)
import bench_v2  # noqa: E402,F401  (v2 benchmarks)
