"""PALOC v2: PALOC + (i) neighbourhood pair scan, (ii) two-output read-out, (iii) native missing channels.

(i)  For every pair of adjacent buses P = {b, b'} the exact linear-Gaussian leave-two-out statistic
         d_P = [Lambda (s - mu)]_P^T (Lambda_PP)^{-1} [Lambda (s - mu)]_P
     is the generalised likelihood ratio for an unknown coordinated shift of the two bus states. It is fused with
     the single-bus neural statistic by a max over log-statistics standardised on training normals.
(ii) Detection uses the fused statistic; localisation uses the anchor (linear LOO) statistic arg-max.
(iii) Missing channels (NaN) receive zero weight in the fixed-gain estimator (physics.ChordSE.estimate).
"""
import numpy as np
import torch

from candidates import BSP, _final_hp
from common import DEV
from registry import register


class PALOC2(BSP):
    supports_missing = True

    def __init__(self, ds, grid, seed=0, pairscan=True, **kw):
        super().__init__(ds, grid, seed, **kw)
        self.pairscan = pairscan
        self.supports_missing = kw.get("state_source", "chord") == "chord"
        # pair index sets over the flattened (valid) standardised state
        valid = self.a_valid.cpu().numpy()
        pos = -np.ones((grid.N, 2), int)
        pos.flat[np.where(valid)[0]] = np.arange(valid.sum())
        P = self.a_P.cpu().numpy()
        pairs = sorted({tuple(sorted((int(f), int(t)))) for f, t in zip(grid.f, grid.t) if f != t})
        idx = np.zeros((len(pairs), 4), int); msk = np.zeros((len(pairs), 4))
        inv = np.zeros((len(pairs), 4, 4))
        for k, (a, b) in enumerate(pairs if pairscan else []):  # pair statistics only needed for the pair-scan variant
            ii = np.r_[pos[a][pos[a] >= 0], pos[b][pos[b] >= 0]]
            idx[k, :len(ii)] = ii; msk[k, :len(ii)] = 1
            inv[k, :len(ii), :len(ii)] = np.linalg.inv(P[np.ix_(ii, ii)])
        self.pairs = np.array(pairs)
        self.p_idx = torch.tensor(idx, device=DEV); self.p_msk = torch.tensor(msk, device=DEV)
        self.p_inv = torch.tensor(inv, device=DEV)
        # standardisation of the two log-statistics on training normals
        Xtr, _ = self.est(ds.normals("train"))
        s1, s2 = self._stats(Xtr)
        l1, l2 = np.log(s1 + 1e-9), np.log(s2 + 1e-9)
        self.c1 = (l1.mean(), l1.std() + 1e-9); self.c2 = (l2.mean(), l2.std() + 1e-9)

    @torch.no_grad()
    def _pair_stats(self, S):
        f = S.flatten(1)[:, self.a_valid].double() - self.a_mu
        r = f @ self.a_P                                        # [B, n]
        rp = r[:, self.p_idx] * self.p_msk                      # [B, Np, 4]
        return torch.einsum("bpi,pij,bpj->bp", rp, self.p_inv, rp)

    @torch.no_grad()
    def _stats(self, X, bs=4096):
        S = torch.tensor(self.rep(X), device=DEV)
        d1, d2 = [], []
        for i in range(0, len(S), bs):
            s = S[i:i + bs]
            D = self._d2(s)[0]
            d1.append((D.max(1).values if self.agg != "sum" else D.sum(1)).double())
            d2.append(self._pair_stats(s).max(1).values if self.pairscan else torch.zeros(len(s), dtype=torch.float64, device=DEV))
        return torch.cat(d1).cpu().numpy(), torch.cat(d2).cpu().numpy()

    def score(self, Z):
        X, _ = self.est(Z)
        s1, s2 = self._stats(X)
        z1 = (np.log(s1 + 1e-9) - self.c1[0]) / self.c1[1]
        if not self.pairscan:
            return z1
        z2 = (np.log(s2 + 1e-9) - self.c2[0]) / self.c2[1]
        return np.maximum(z1, z2)

    @torch.no_grad()
    def loc_stats(self, Z):
        """Localisation read-out: whitened linear LOO statistic per bus."""
        X, _ = self.est(Z)
        S = torch.tensor(self.rep(X), device=DEV)
        e = self.target(S)
        w = torch.einsum("nij,bnj->bni", self.Lc_inv, e)
        return ((w * self.dimmask) ** 2).sum(-1).cpu().numpy()

    def bus_stats(self, Z):  # used by the localisation script: two-output design -> anchor read-out
        return self.loc_stats(Z), None


def _make(pairscan, label_free=False):
    def b(ds, grid, seed=0, **kw):
        hp = dict(_final_hp())
        if label_free:  # pre-specified defaults, no use of labelled validation attacks
            hp.update(lr=2e-3, wd=1e-4, agg="max", fuse=False, r=64, hidden=128, depth=2, emb=8)
        hp.update(kw)
        return PALOC2(ds, grid, seed, pairscan=pairscan, **hp)
    return b


register("PALOC2")(_make(False))          # final v2 model (pair scan not retained: no validation gain)
register("PALOC2-pair")(_make(True))      # validation-stage extension (selection record only)
register("PALOC2-nopair")(_make(False))
register("PALOC2-labelfree")(_make(False, label_free=True))
for _name, _over in {"PALOC2-noNN": {"neural": False}, "PALOC2-sum": {"agg": "sum"}, "PALOC2-learnedSE": {"state_source": "learned"},
                     "PALOC2-noAnchor": {"anchor": False}, "PALOC2-diag": {"diag": True}, "PALOC2-fullWLS": {"state_source": "wls"}}.items():
    register(_name)((lambda o: (lambda ds, grid, seed=0, **kw: PALOC2(ds, grid, seed, pairscan=False, **{**_final_hp(), **o, **kw})))(_over))
