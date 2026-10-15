"""Privileged supervised reference detectors (trained WITH synthetic attack labels from the TRAIN split:
train normals + train_mix attacks; validation normals + val_mix attacks for early stopping).
They are reported separately from the unsupervised comparison because they see attack labels.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from common import DEV
from registry import register, n_params


def xy(ds, split, bus_filter=None):
    Zn = ds.normals(split)
    key = "train_mix" if split == "train" else "val_mix"
    Za, b = ds.att(key), ds.att(key, "bus")
    if bus_filter is not None:
        Za = Za[np.isin(b, bus_filter)]
    X = ds.norm(np.r_[Zn, Za])
    y = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]
    return X, y


class SupDet:
    def __init__(self, ds, fit_fn, bus_filter=None):
        self.ds = ds
        Xtr, ytr = xy(ds, "train", bus_filter); Xv, yv = xy(ds, "val", bus_filter)
        self.est, self.extra = fit_fn(Xtr, ytr, Xv, yv)

    def score(self, Z):
        X = self.ds.norm(Z)
        if hasattr(self.est, "predict_proba"):
            p = self.est.predict_proba(X)[:, 1]
            return np.log(p + 1e-12) - np.log(1 - p + 1e-12)
        return self.est(X)

    def info(self):
        return self.extra or {}


def _wrap(name, fit_fn):
    @register(name)
    def _b(ds, grid, seed=0, bus_filter=None, **kw):
        return SupDet(ds, lambda *a: fit_fn(*a, seed=seed, **kw), bus_filter)
    return _b


_wrap("Sup-LR", lambda X, y, Xv, yv, seed=0, C=1.0: (LogisticRegression(C=C, max_iter=3000).fit(X, y), {}))
_wrap("Sup-RF", lambda X, y, Xv, yv, seed=0: (RandomForestClassifier(500, n_jobs=8, random_state=seed).fit(X, y), {}))
_wrap("Sup-ET", lambda X, y, Xv, yv, seed=0: (ExtraTreesClassifier(500, n_jobs=8, random_state=seed).fit(X, y), {}))
_wrap("Sup-HGB", lambda X, y, Xv, yv, seed=0: (HistGradientBoostingClassifier(max_iter=500, early_stopping=True, random_state=seed).fit(X, y), {}))


def _xgb(X, y, Xv, yv, seed=0, **kw):
    import xgboost as xgb
    p = dict(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5,
             tree_method="hist", device=("cuda" if DEV == "cuda" else "cpu"), n_jobs=8, early_stopping_rounds=50, random_state=seed, eval_metric="logloss")
    p.update(kw)
    m = xgb.XGBClassifier(**p).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    return m, {"n_trees": m.best_iteration + 1}


def _lgbm(X, y, Xv, yv, seed=0, **kw):
    import lightgbm as lgb
    p = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.5, random_state=seed, verbose=-1, n_jobs=8)
    p.update(kw)
    m = lgb.LGBMClassifier(**p).fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, {"n_trees": m.best_iteration_}


def _cat(X, y, Xv, yv, seed=0, **kw):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, random_seed=seed, verbose=0,
                           task_type=("GPU" if DEV == "cuda" else "CPU"), thread_count=8, od_type="Iter", od_wait=50, **kw).fit(X, y, eval_set=(Xv, yv))
    return m, {"n_trees": m.get_best_iteration() + 1}


_wrap("Sup-XGB", _xgb)
_wrap("Sup-LGBM", _lgbm)
_wrap("Sup-CatBoost", _cat)


# ------------------------------------------------------------------ deep supervised
class FTTransformer(nn.Module):
    """Feature-tokenizer Transformer (Gorishniy et al., 2021). Channels grouped into tokens
    of size `patch` to keep sequence length tractable on 118-bus (1096 channels)."""

    def __init__(self, m, d=64, layers=3, heads=4, patch=8):
        super().__init__()
        self.patch = patch
        self.T = int(np.ceil(m / patch))
        self.pad = self.T * patch - m
        self.tok = nn.Parameter(torch.randn(self.T, patch, d) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.T, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        x = nn.functional.pad(x, (0, self.pad)).view(len(x), self.T, self.patch)
        t = torch.einsum("btp,tpd->btd", x, self.tok) + self.bias
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        return self.head(self.tr(t)[:, 0]).squeeze(-1)


class MLPc(nn.Module):
    def __init__(self, m, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(m, h), nn.SiLU(), nn.Dropout(0.1), nn.Linear(h, h // 2), nn.SiLU(),
                                 nn.Dropout(0.1), nn.Linear(h // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GNNc(nn.Module):
    def __init__(self, grid, d=64, layers=4):
        super().__init__()
        from candidates import GraphEncoder
        self.enc = GraphEncoder(grid, d, layers)
        self.enc.out = nn.Identity()
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 1))
        self.g = grid

    def forward(self, x):
        # reuse encoder trunk: returns concatenated per-node features via a forward hook-free path
        enc = self.enc
        B, N = x.shape[0], self.g.N
        h0 = torch.zeros(B, N, 3, device=x.device)
        h0[..., 0] = x[:, enc.iV]; h0[:, enc.nr, 1] = x[:, enc.iP]; h0[:, enc.nr, 2] = x[:, enc.iQ]
        fl = x[:, enc.iF].permute(0, 2, 1)
        e = enc.einp(torch.cat([fl, fl[..., [2, 3, 0, 1]]], 1))
        h = enc.inp(h0)
        for msg, upd, ln in zip(enc.msg, enc.upd, enc.ln):
            m = nn.functional.silu(msg(torch.cat([h[:, enc.src], h[:, enc.dst], e], -1)))
            agg = torch.zeros_like(h).index_add_(1, enc.dst, m)
            h = ln(h + upd(torch.cat([h, agg], -1)))
        return self.head(torch.cat([h.mean(1), h.max(1).values], -1)).squeeze(-1)


def _torch_fit(model, X, y, Xv, yv, seed=0, epochs=100, lr=1e-3, bs=256, patience=10):
    torch.manual_seed(seed)
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.tensor(X, device=DEV), torch.tensor(y, device=DEV, dtype=torch.float32)
    Xvt, yvt = torch.tensor(Xv, device=DEV), torch.tensor(yv, device=DEV, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss()
    best, bad, bst = np.inf, 0, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        p = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(p), bs):
            l = lossf(model(Xt[p[i:i + bs]]), yt[p[i:i + bs]])
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([lossf(model(Xvt[i:i + 1024]), yvt[i:i + 1024]).item() for i in range(0, len(Xvt), 1024)])
        if vl < best - 1e-5:
            best, bad, bst = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(bst); model.eval()

    @torch.no_grad()
    def f(Xn):
        return torch.cat([model(torch.tensor(Xn[i:i + 2048], device=DEV)) for i in range(0, len(Xn), 2048)]).cpu().numpy()
    return f, {"params": n_params(model), "epochs": ep + 1, "train_time": time.time() - t0}


@register("Sup-MLP")
def _smlp(ds, grid, seed=0, bus_filter=None, **kw):
    return SupDet(ds, lambda X, y, Xv, yv: _torch_fit(MLPc(X.shape[1]), X, y, Xv, yv, seed), bus_filter)


@register("Sup-FTT")
def _sftt(ds, grid, seed=0, bus_filter=None, **kw):
    return SupDet(ds, lambda X, y, Xv, yv: _torch_fit(FTTransformer(X.shape[1]), X, y, Xv, yv, seed, lr=3e-4), bus_filter)


@register("Sup-GNN")
def _sgnn(ds, grid, seed=0, bus_filter=None, **kw):
    return SupDet(ds, lambda X, y, Xv, yv: _torch_fit(GNNc(grid), X, y, Xv, yv, seed), bus_filter)
