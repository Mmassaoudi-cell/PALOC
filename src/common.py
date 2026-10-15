"""Shared utilities: dataset access, train-only preprocessing, evaluation sets, metrics."""
import os
import random
import re

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
SYSTEMS = ["ieee14", "ieee39", "ieee118"]
TIERS = ["S", "M", "W"]
DEV = os.environ.get("RUN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Dataset:
    """One (system, replicate). All preprocessing statistics come from TRAIN normals only."""

    def __init__(self, system, rep):
        self.system, self.rep = system, rep
        self.base = re.match(r"(ieee\d+)", system).group(1)
        d = np.load(os.path.join(DATA_DIR, f"{system}_rep{rep}.npz"), allow_pickle=False)
        self.d = d
        self.X, self.Z, self.split, self.J = d["X"], d["Z"], d["split"], d["J"]
        self.idx = {s: np.where(self.split == s)[0] for s in ("train", "val", "test")}
        tr = self.Z[self.idx["train"]].astype(np.float64)
        self.mu = tr.mean(0)
        self.sd = tr.std(0) + 1e-8

    def att(self, key, field="Z"):
        return self.d[f"A_{key}_{field}"]

    def norm(self, Z):
        return ((Z - self.mu) / self.sd).astype(np.float32)

    def normals(self, split):
        return self.Z[self.idx[split]]

    def eval_sets(self):
        """Dict of raw-measurement matrices to score: normals + attack sets."""
        S = {"train_N": self.normals("train"), "val_N": self.normals("val"), "test_N": self.normals("test"),
             "val_A": self.att("val_mix")}
        vt = self.att("val_mix", "tier")
        for t in TIERS:
            S[f"val_A_{t}"] = S["val_A"][vt == t]
        for t in TIERS:
            S[f"test_A_{t}"] = self.att(f"test_{t}")
        S["test_A_generic"] = self.att("test_generic")
        S["test_A_gray"] = self.att("test_gray")
        if self.system.endswith("p2"):  # robustness sets (P2 = primary protocol)
            from grid import Grid
            import robust_sets
            for k, v in robust_sets.build(self, Grid(self.system)).items():
                if not k.endswith("_bus"):
                    S[k] = v
        return S


def threshold(val_scores, alpha=0.05):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest calibration score."""
    s = np.sort(np.asarray(val_scores))
    n = len(s)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return s[k - 1]


def metrics(s_norm, s_att, tau):
    y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_att))]
    s = np.r_[s_norm, s_att]
    s = np.nan_to_num(s, nan=np.nanmax(s[np.isfinite(s)]) if np.isfinite(s).any() else 0, posinf=1e30)
    pred = (s > tau).astype(int)
    tpr = pred[y == 1].mean(); fpr = pred[y == 0].mean()
    rec_n = 1 - fpr
    return dict(auc=roc_auc_score(y, s), prauc=average_precision_score(y, s),
                macro_f1=f1_score(y, pred, average="macro"), weighted_f1=f1_score(y, pred, average="weighted"),
                bal_acc=balanced_accuracy_score(y, pred), tpr=tpr, fpr=fpr, fnr=1 - tpr,
                recall_normal=rec_n, recall_attack=tpr,
                f1_attack=f1_score(y, pred), f1_normal=f1_score(1 - y, 1 - pred))


def evaluate_scores(sc, alphas=(0.05, 0.01)):
    """sc: dict of score arrays from eval_sets keys. Returns list of metric rows."""
    rows = []
    for a in alphas:
        tau = threshold(sc["val_N"], a)
        rows.append(dict(set="val_mix", alpha=a, **metrics(sc["val_N"], sc["val_A"], tau)))
        for k in sc:
            if k.startswith("val_A_"):
                rows.append(dict(set="val_" + k[6:], alpha=a, **metrics(sc["val_N"], sc[k], tau)))
            if k.startswith("test_A_"):
                rows.append(dict(set=k[7:], alpha=a, **metrics(sc["test_N"], sc[k], tau)))
            if k.startswith("rob_") and "_A_" in k:
                rows.append(dict(set=k, alpha=a, **metrics(sc[k.split("_A_")[0] + "_N"], sc[k], tau)))
            if k.startswith("sweep"):
                rows.append(dict(set=k, alpha=a, **metrics(sc["test_N"], sc[k], tau)))
    return rows


def bus_groups(grid):
    """For each bus, indices of measurement channels associated with it (V, injections, incident flows)."""
    groups = [[] for _ in range(grid.N)]
    for c, (b, br) in enumerate(zip(grid.ch_bus, grid.ch_branch)):
        groups[b].append(c)
        if br >= 0:  # flows are also attributed to the other end of the branch
            other = grid.t[br] if grid.f[br] == b else grid.f[br]
            groups[other].append(c)
    return [np.array(sorted(set(g))) for g in groups]
