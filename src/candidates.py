"""Candidate hybrid detectors (see MODEL_CANDIDATES.md).

C1  PGMP   : chord-WLS physics state -> masked-bus conditional Gaussian prior (GNN or MLP backbone)
             -> bus-localised Mahalanobis statistics -> max pooling (+ optional chi-square fusion)
C2  PIBAEHL: source PI-BAE architecture + per-channel heteroscedastic likelihood + bus-group max pooling
C3  PSAEH  : chord-WLS physics state -> global state autoencoder with heteroscedastic output -> bus max
C4  LGNNMP : learned graph encoder (amortised state inversion, physics loss) -> masked-bus prior
Also: LocalGGM (linear-Gaussian closed-form per-bus conditional test; used as an ablation).
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf

import ae_models as A
from common import DEV
from physics import ChordSE
from registry import register, n_params


# ============================================================================ helpers
def colorings(grid, n_random=16, seed=0):
    """Greedy distance-1 colourings (independent sets). [0] is deterministic (used at inference)."""
    A_ = grid.adjacency()
    rng = np.random.default_rng(seed)
    outs = []
    for r in range(n_random + 1):
        order = np.argsort(-A_.sum(1), kind="stable") if r == 0 else rng.permutation(grid.N)
        col = -np.ones(grid.N, int)
        for v in order:
            used = set(col[A_[v]])
            c = 0
            while c in used:
                c += 1
            col[v] = c
        outs.append(col)
    return outs


def masks_from_colorings(cols, N):
    """All colour classes of all colourings as a boolean mask bank [K, N]."""
    bank = []
    for col in cols:
        for c in range(col.max() + 1):
            bank.append(col == c)
    return np.array(bank)


class StateRep:
    """Per-bus standardised state representation [B, N, 2] = (V, theta); stats from TRAIN normals."""

    def __init__(self, grid, Xtr):
        self.g = grid
        V, th = self.split(Xtr)
        self.mv, self.sv = V.mean(0), V.std(0) + 1e-6
        self.mt, self.st = th.mean(0), th.std(0) + 1e-6
        self.st[grid.ref] = 1.0
        self.dimmask = np.ones((grid.N, 2), np.float32); self.dimmask[grid.ref, 1] = 0.0

    def split(self, X):
        g = self.g
        V = X[:, : g.N]
        th = np.zeros_like(V); th[:, g.nonref] = X[:, g.N:]
        return V, th

    def __call__(self, X):
        V, th = self.split(X)
        return np.stack([(V - self.mv) / self.sv, (th - self.mt) / self.st], -1).astype(np.float32)


class MaskedPrior(nn.Module):
    """Predicts a bivariate Gaussian for each masked bus from the unmasked buses' states."""

    def __init__(self, grid, backbone="gnn", d=48, layers=3, hidden=256, bus_emb=8, dropout=0.0):
        super().__init__()
        self.N, self.backbone = grid.N, backbone
        types = np.zeros((grid.N, 3), np.float32)
        bt = grid.net._ppc["bus"][:, 1].real.astype(int)
        types[np.arange(grid.N), np.clip(bt - 1, 0, 2)] = 1
        self.register_buffer("types", torch.tensor(types))
        self.emb = nn.Parameter(0.1 * torch.randn(grid.N, bus_emb)) if bus_emb > 0 else None
        fin = 3 + 3 + (bus_emb if bus_emb > 0 else 0)
        if backbone == "gnn":
            src = np.r_[grid.f, grid.t]; dst = np.r_[grid.t, grid.f]
            y = -grid.Ybus[grid.f, grid.t]
            ef = np.stack([y.real, y.imag], 1)
            ef = (ef - ef.mean(0)) / (ef.std(0) + 1e-6)
            self.register_buffer("src", torch.tensor(src)); self.register_buffer("dst", torch.tensor(dst))
            self.register_buffer("ef", torch.tensor(np.r_[ef, ef], dtype=torch.float32))
            self.inp = nn.Linear(fin, d)
            self.msg = nn.ModuleList([nn.Linear(2 * d + 2, d) for _ in range(layers)])
            self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d)) for _ in range(layers)])
            self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 5))
        else:  # global masked MLP
            self.net = nn.Sequential(nn.Linear(grid.N * 3, hidden), nn.SiLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, grid.N * 5))

    def forward(self, s, mask):
        """s [B,N,2] standardised states, mask [B,N] bool (True = hidden). Returns mu [B,N,2], L params."""
        B = s.shape[0]
        m = mask.float().unsqueeze(-1)
        x = torch.cat([s * (1 - m), m], -1)  # [B,N,3]
        if self.backbone == "gnn":
            feats = [x, self.types.expand(B, -1, -1)]
            if self.emb is not None:
                feats.append(self.emb.expand(B, -1, -1))
            h = self.inp(torch.cat(feats, -1))
            for msg, upd, ln in zip(self.msg, self.upd, self.ln):
                e = msg(torch.cat([h[:, self.src], h[:, self.dst], self.ef.expand(B, -1, -1)], -1))
                agg = torch.zeros_like(h).index_add_(1, self.dst, F.silu(e))
                h = ln(h + upd(torch.cat([h, agg], -1)))
            o = self.head(h)
        else:
            o = self.net(x.flatten(1)).view(B, self.N, 5)
        mu = o[..., :2]
        l11 = F.softplus(o[..., 2]) + 1e-3
        l22 = F.softplus(o[..., 3]) + 1e-3
        l21 = o[..., 4]
        return mu, l11, l22, l21


def gauss2_d2(t, mu, l11, l22, l21, dimmask):
    """Squared Mahalanobis distance and log-det for Sigma = L L^T (lower-triangular L)."""
    r = (t - mu) * dimmask
    u1 = r[..., 0] / l11
    u2 = (r[..., 1] - l21 * u1) / l22 * dimmask[..., 1]
    d2 = u1 ** 2 + u2 ** 2
    logdet = torch.log(l11) + torch.log(l22) * dimmask[..., 1]
    return d2, logdet


def aggregate(D, how):
    if how == "max":
        return D.max(1)
    if how == "top3":
        return np.sort(D, 1)[:, -3:].mean(1)
    if how == "sum":
        return D.sum(1)
    if how == "lse":
        return np.log(np.exp(D - D.max(1, keepdims=True)).sum(1)) + D.max(1)
    raise ValueError(how)


class FusedScore:
    """Max-fusion of the localised prior statistic and the physics chi-square residual.
    Both standardised: prior on a held-out TRAIN calibration slice, J by its chi-square law."""

    def __init__(self, s_cal, dof):
        self.log = bool(np.all(s_cal > 0))
        ls = np.log(s_cal + 1e-12) if self.log else s_cal
        self.m, self.s, self.dof = ls.mean(), ls.std() + 1e-9, dof

    def __call__(self, s, J):
        zp = ((np.log(s + 1e-12) if self.log else s) - self.m) / self.s
        zj = (J - self.dof) / np.sqrt(2 * self.dof)
        return np.maximum(zp, zj)


# ============================================================================ C1 PGMP
class PGMP:
    def __init__(self, ds, grid, seed=0, backbone="gnn", d=48, layers=3, hidden=256, bus_emb=8,
                 lr=2e-3, wd=1e-5, epochs=150, bs=128, patience=20, agg="max", fuse=False,
                 se_iters=2, state_source="chord", encoder=None, dropout=0.0, cal_frac=0.1, anchor=False):
        self.g, self.agg, self.fuse, self.anchor = grid, agg, fuse, anchor
        t0 = time.time()
        if state_source == "chord":
            self.se = ChordSE(grid, iters=se_iters)
            est = lambda Z: self.se.estimate(Z)
        else:  # learned encoder (C4)
            self.se = None
            est = encoder
        self.est = est
        Ztr = ds.normals("train")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Ztr))
        ncal = int(cal_frac * len(Ztr)) if (fuse or agg == "maxcal") else 0
        cal, fit = perm[:ncal], perm[ncal:]
        Xtr, Jtr = est(Ztr)
        Xv, _ = est(ds.normals("val"))
        self.rep = StateRep(grid, Xtr[fit])
        self.se_time = time.time() - t0
        Str = torch.tensor(self.rep(Xtr[fit]), device=DEV); Sv = torch.tensor(self.rep(Xv), device=DEV)
        self.dimmask = torch.tensor(self.rep.dimmask, device=DEV)
        if anchor:
            self._fit_anchor(Str)
        cols = colorings(grid, seed=seed)
        self.bank = torch.tensor(masks_from_colorings(cols, grid.N), device=DEV)
        self.infer_masks = torch.tensor(masks_from_colorings(cols[:1], grid.N), device=DEV)
        torch.manual_seed(seed)
        self.model = MaskedPrior(grid, backbone, d, layers, hidden, bus_emb, dropout).to(DEV)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, bad, best_state = np.inf, 0, None
        t1 = time.time()
        for ep in range(epochs):
            self.model.train()
            p = torch.randperm(len(Str), device=DEV)
            for i in range(0, len(p), bs):
                s = Str[p[i:i + bs]]
                mk = self.bank[torch.randint(len(self.bank), (len(s),), device=DEV)]
                loss = self._nll(s, mk)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            sched.step()
            self.model.eval()
            with torch.no_grad():
                vl = np.mean([self._nll(Sv, self.infer_masks[k].expand(len(Sv), -1)).item()
                              for k in range(len(self.infer_masks))])
            if vl < best - 1e-5:
                best, bad, best_state = vl, 0, copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        self.model.load_state_dict(best_state); self.model.eval()
        self.train_time, self.epochs = time.time() - t1, ep + 1
        self._post_fit(Xtr[cal], fuse)

    def _post_fit(self, Xcal, fuse):
        """Per-bus calibration (median / IQR of log d_b on a held-out TRAIN slice) and optional chi^2 fusion."""
        self.fuser, self.bcal = None, None
        if len(Xcal) == 0:
            return
        Dc = self.bus_stats_from_states(Xcal)
        if self.agg == "maxcal":
            L = np.log(Dc + 1e-9)
            q25, q50, q75 = np.percentile(L, [25, 50, 75], axis=0)
            self.bcal = (q50, np.maximum(q75 - q25, 1e-6))
        if fuse:
            self.fuser = FusedScore(self._agg(Dc), self.g.m - self.g.n)

    def _agg(self, D):
        if self.agg == "maxcal":
            return ((np.log(D + 1e-9) - self.bcal[0]) / self.bcal[1]).max(1)
        return aggregate(D, self.agg)

    def _fit_anchor(self, S):
        """Closed-form linear-Gaussian leave-one-bus-out conditional (Ledoit-Wolf precision on TRAIN states)."""
        valid = self.dimmask.flatten().bool()
        F_ = S.flatten(1)[:, valid].double().cpu().numpy()
        self.a_mu = torch.tensor(F_.mean(0), device=DEV)
        P = LedoitWolf().fit(F_ - F_.mean(0)).precision_
        self.a_P = torch.tensor(P, device=DEV)
        self.a_valid = valid
        pos = -np.ones((self.g.N, 2), int); pos.flat[np.where(valid.cpu().numpy())[0]] = np.arange(valid.sum().item())
        Binv = np.zeros((self.g.N, 2, 2))
        for b in range(self.g.N):
            ii = pos[b][pos[b] >= 0]
            Binv[b, :len(ii), :len(ii)] = np.linalg.inv(P[np.ix_(ii, ii)])
        pos[pos < 0] = 0
        self.a_pos = torch.tensor(pos, device=DEV); self.a_Binv = torch.tensor(Binv, device=DEV)

    def target(self, s):
        """Anchored target: e_b = s_b - E_lin[s_b | s_-b] (exact LOO; uses no information about bus b's own
        conditional beyond its own value), else the raw standardised state."""
        if not self.anchor:
            return s
        f = s.flatten(1)[:, self.a_valid].double() - self.a_mu
        r = f @ self.a_P  # [B, n]
        rb = r[:, self.a_pos] * self.dimmask  # [B, N, 2]
        e = torch.einsum("nij,bnj->bni", self.a_Binv, rb)
        return e.float()

    def _nll(self, s, mk):
        mu, l11, l22, l21 = self.model(s, mk)
        d2, ld = gauss2_d2(self.target(s), mu, l11, l22, l21, self.dimmask)
        w = mk.float()
        return ((0.5 * d2 + ld) * w).sum() / w.sum()

    @torch.no_grad()
    def bus_stats_from_states(self, X, bs=4096):
        S = torch.tensor(self.rep(X), device=DEV)
        out = torch.zeros(S.shape[0], self.g.N, device=DEV)
        for i in range(0, len(S), bs):
            s = S[i:i + bs]
            for mk in self.infer_masks:
                mu, l11, l22, l21 = self.model(s, mk.expand(len(s), -1))
                d2, _ = gauss2_d2(self.target(s), mu, l11, l22, l21, self.dimmask)
                out[i:i + bs, mk] = d2[:, mk]
        return out.cpu().numpy()

    def bus_stats(self, Z):
        X, J = self.est(Z)
        return self.bus_stats_from_states(X), J

    def score(self, Z):
        D, J = self.bus_stats(Z)
        s = self._agg(D)
        return self.fuser(s, J) if self.fuser is not None else s

    def info(self):
        p = n_params(self.model) if getattr(self, "neural", True) else 0
        return dict(params=p, epochs=self.epochs, train_time=self.train_time,
                    n_mask_passes=len(self.infer_masks))


@register("PGMP")
def _pgmp(ds, grid, seed=0, **kw):
    return PGMP(ds, grid, seed, **kw)


@register("PGMP_A")
def _pgmp_a(ds, grid, seed=0, **kw):
    kw.setdefault("anchor", True)
    return PGMP(ds, grid, seed, **kw)


@register("PGMP_AM")
def _pgmp_am(ds, grid, seed=0, **kw):
    kw.setdefault("anchor", True); kw.setdefault("backbone", "mlp")
    return PGMP(ds, grid, seed, **kw)


@register("PGMP_MLP")
def _pgmp_mlp(ds, grid, seed=0, **kw):
    kw.setdefault("backbone", "mlp")
    return PGMP(ds, grid, seed, **kw)


# ============================================================================ LocalGGM (linear ablation)
class LocalGGM:
    """Closed-form linear-Gaussian per-bus conditional test on chord-WLS states:
    r = Lambda (x - mu); d_b = r_b^T (Lambda_bb)^{-1} r_b  (conditional Mahalanobis of bus b given rest)."""

    def __init__(self, ds, grid, seed=0, se_iters=2, agg="max", **_):
        self.g, self.agg = grid, agg
        self.se = ChordSE(grid, iters=se_iters)
        Xtr, _ = self.se.estimate(ds.normals("train"))
        self.mu = Xtr.mean(0)
        self.P = LedoitWolf().fit(Xtr - self.mu).precision_
        n, N = grid.n, grid.N
        self.idx = []
        for b in range(N):
            ii = [b] + ([N + int(np.where(grid.nonref == b)[0][0])] if b != grid.ref else [])
            self.idx.append(np.array(ii))
        self.Pinv_bb = [np.linalg.inv(self.P[np.ix_(ii, ii)]) for ii in self.idx]

    def score(self, Z):
        X, _ = self.se.estimate(Z)
        R = (X - self.mu) @ self.P
        D = np.stack([np.einsum("bi,ij,bj->b", R[:, ii], Pi, R[:, ii]) for ii, Pi in zip(self.idx, self.Pinv_bb)], 1)
        return aggregate(D, self.agg)


register("LocalGGM")(LocalGGM)


# ============================================================================ C2 PIBAE-HL
class PIBAEHL(A.PIAE):
    """PI-BAE with per-channel heteroscedastic Gaussian likelihoods and bus-group max pooling."""

    def __init__(self, m, n, widths, phys, groups, agg="max"):
        super().__init__(m, n, widths, phys)
        self.ls1 = nn.Parameter(torch.zeros(m)); self.ls2 = nn.Parameter(torch.zeros(m))
        G = np.zeros((len(groups), m), np.float32)
        for b, gidx in enumerate(groups):
            G[b, gidx] = 1.0 / len(gidx)
        self.register_buffer("G", torch.tensor(G))
        self.agg = agg

    def chan(self, z):
        u = self.E(z)
        es = (z - self.D(u)) ** 2; ep = (z - self.P(u)) ** 2
        return es, ep

    def loss(self, z):
        es, ep = self.chan(z)
        return (es / (2 * torch.exp(2 * self.ls1)) + self.ls1 + ep / (2 * torch.exp(2 * self.ls2)) + self.ls2).mean()

    def score(self, z):
        es, ep = self.chan(z)
        c = es / torch.exp(2 * self.ls1) + ep / torch.exp(2 * self.ls2)
        Gs = c @ self.G.T  # [B, N]
        if self.agg == "max":
            return Gs.max(1).values
        return c.mean(1)


@register("PIBAEHL")
def _pibaehl(ds, grid, seed=0, agg="max", **kw):
    from common import bus_groups
    m = PIBAEHL(grid.m, grid.n, A.WIDTHS[ds.base], A.PhysLayer(grid, ds.mu, ds.sd), bus_groups(grid), agg)
    from registry import AEDet
    return AEDet(ds, m, **kw)


# ============================================================================ C3 PSAE-H
class StateAEH(nn.Module):
    def __init__(self, N, k=16, hidden=256):
        super().__init__()
        self.N = N
        self.E = nn.Sequential(nn.Linear(2 * N, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, k))
        self.D = nn.Sequential(nn.Linear(k, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4 * N))

    def forward(self, s):
        o = self.D(self.E(s.flatten(1))).view(len(s), self.N, 4)
        return o[..., :2], o[..., 2:].clamp(-8, 8)


class PSAEH:
    def __init__(self, ds, grid, seed=0, k=16, hidden=256, lr=1e-3, epochs=200, bs=128, patience=20, agg="max", se_iters=2):
        self.g, self.agg = grid, agg
        self.se = ChordSE(grid, iters=se_iters)
        Xtr, _ = self.se.estimate(ds.normals("train")); Xv, _ = self.se.estimate(ds.normals("val"))
        self.rep = StateRep(grid, Xtr)
        self.dm = torch.tensor(self.rep.dimmask, device=DEV)
        Str = torch.tensor(self.rep(Xtr), device=DEV); Sv = torch.tensor(self.rep(Xv), device=DEV)
        torch.manual_seed(seed)
        self.model = StateAEH(grid.N, k, hidden).to(DEV)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        best, bad, bst = np.inf, 0, None
        t0 = time.time()
        for ep in range(epochs):
            self.model.train()
            p = torch.randperm(len(Str), device=DEV)
            for i in range(0, len(p), bs):
                l = self._nll(Str[p[i:i + bs]])
                opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = self._nll(Sv).item()
            if vl < best - 1e-5:
                best, bad, bst = vl, 0, copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        self.model.load_state_dict(bst); self.model.eval()
        self.train_time, self.epochs = time.time() - t0, ep + 1

    def _d(self, s):
        mu, lv = self.model(s)
        return ((s - mu) ** 2 / torch.exp(lv)) * self.dm, lv * self.dm

    def _nll(self, s):
        d, lv = self._d(s)
        return (0.5 * d + 0.5 * lv).sum((1, 2)).mean() / self.g.N

    @torch.no_grad()
    def score(self, Z):
        X, _ = self.se.estimate(Z)
        d, _ = self._d(torch.tensor(self.rep(X), device=DEV))
        return aggregate(d.sum(-1).cpu().numpy(), self.agg)

    def info(self):
        return dict(params=n_params(self.model), epochs=self.epochs, train_time=self.train_time)


@register("PSAEH")
def _psaeh(ds, grid, seed=0, **kw):
    return PSAEH(ds, grid, seed, **kw)


# ============================================================================ C4 LGNN-MP
class GraphEncoder(nn.Module):
    """Weight-shared message-passing encoder: per-bus measurements -> per-bus (V, theta) state."""

    def __init__(self, grid, d=64, layers=4):
        super().__init__()
        self.g = grid
        N, L = grid.N, grid.L
        # node inputs: V, P, Q (0 at ref); edge inputs: Pf, Qf, Pt, Qt
        self.register_buffer("iV", torch.arange(N))
        nr = grid.nonref
        self.register_buffer("iP", torch.tensor(N + np.arange(N - 1))); self.register_buffer("iQ", torch.tensor(2 * N - 1 + np.arange(N - 1)))
        self.register_buffer("nr", torch.tensor(nr))
        base = N + 2 * (N - 1)
        self.register_buffer("iF", torch.tensor(base + np.arange(4 * L)).view(4, L))
        self.register_buffer("src", torch.tensor(np.r_[grid.f, grid.t])); self.register_buffer("dst", torch.tensor(np.r_[grid.t, grid.f]))
        self.inp = nn.Linear(3, d)
        self.einp = nn.Linear(4, d)
        self.msg = nn.ModuleList([nn.Linear(3 * d, d) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d)) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.out = nn.Linear(d, 2)

    def forward(self, zn):
        B, N = zn.shape[0], self.g.N
        x = torch.zeros(B, N, 3, device=zn.device)
        x[..., 0] = zn[:, self.iV]
        x[:, self.nr, 1] = zn[:, self.iP]; x[:, self.nr, 2] = zn[:, self.iQ]
        fl = zn[:, self.iF].permute(0, 2, 1)  # [B, L, 4]
        fl_rev = fl[..., [2, 3, 0, 1]]
        e = self.einp(torch.cat([fl, fl_rev], 1))
        h = self.inp(x)
        for msg, upd, ln in zip(self.msg, self.upd, self.ln):
            m = F.silu(msg(torch.cat([h[:, self.src], h[:, self.dst], e], -1)))
            agg = torch.zeros_like(h).index_add_(1, self.dst, m)
            h = ln(h + upd(torch.cat([h, agg], -1)))
        o = self.out(h)  # [B,N,2]
        u = torch.cat([o[..., 0], o[:, self.nr, 1]], 1)
        return u


class LGNNEnc(A.Base):
    def __init__(self, grid, phys, d=64, layers=4):
        super().__init__()
        self.E = GraphEncoder(grid, d, layers)
        self.P = phys

    def loss(self, z):
        return ((z - self.P(self.E(z))) ** 2).mean()


@register("LGNNMP")
def _lgnnmp(ds, grid, seed=0, enc_d=64, enc_layers=4, **kw):
    enc = LGNNEnc(grid, A.PhysLayer(grid, ds.mu, ds.sd), enc_d, enc_layers)
    enc, _ = A.train(enc, ds.norm(ds.normals("train")), ds.norm(ds.normals("val")), epochs=kw.pop("enc_epochs", 200))

    @torch.no_grad()
    def est(Z):
        zn = torch.tensor(ds.norm(Z), device=DEV)
        X = torch.cat([enc.P.to_state(enc.E(zn[i:i + 4096])) for i in range(0, len(zn), 4096)])
        r = torch.cat([((zn[i:i + 4096] - enc.P(enc.E(zn[i:i + 4096]))) ** 2).sum(1) for i in range(0, len(zn), 4096)])
        return X.double().cpu().numpy(), r.cpu().numpy()
    det = PGMP(ds, grid, seed, state_source="learned", encoder=est, **kw)
    det.enc_params = n_params(enc)
    return det


# ============================================================================ C5 BSP (blind-spot prior)
class BlindSpotNet(nn.Module):
    """Exact leave-one-bus-out conditional network.
    g = P^T s (global projection);  u_b = g - P_b^T s_b  (bus b's own state removed exactly)
    head_b = MLP([u_b, lin_b, emb_b]) -> (delta_b in R^2, Cholesky of 2x2 covariance)."""

    def __init__(self, N, r=64, hidden=128, emb=8, depth=2):
        super().__init__()
        self.N = N
        self.P = nn.Parameter(torch.randn(N, 2, r) * (1.0 / np.sqrt(2 * N)))
        self.emb = nn.Parameter(0.1 * torch.randn(N, emb))
        layers, fin = [], r + 2 + emb
        for _ in range(depth):
            layers += [nn.Linear(fin, hidden), nn.SiLU()]
            fin = hidden
        layers.append(nn.Linear(fin, 5))
        self.head = nn.Sequential(*layers)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, s, lin):
        """s [B,N,2] standardised state (masked dims zeroed), lin [B,N,2] linear-LOO prediction of s_b."""
        own = torch.einsum("bnk,nkr->bnr", s, self.P)       # P_b^T s_b
        g = own.sum(1, keepdim=True)                          # P^T s
        u = g - own                                           # excludes bus b exactly
        o = self.head(torch.cat([u, lin, self.emb.expand(len(s), -1, -1)], -1))
        mu = o[..., :2]
        l11 = F.softplus(o[..., 2] + 0.5413) + 1e-3           # softplus(0.5413) = 1 -> identity at init
        l22 = F.softplus(o[..., 3] + 0.5413) + 1e-3
        return mu, l11, l22, o[..., 4]


class BSP(PGMP):
    """C5: chord-WLS state -> linear-Gaussian LOO anchor -> blind-spot neural residual prior -> bus max."""

    def __init__(self, ds, grid, seed=0, r=64, hidden=128, emb=8, depth=2, lr=2e-3, wd=1e-4, epochs=150, bs=128,
                 patience=20, agg="max", se_iters=2, whiten=True, fuse=False, cal_frac=0.1,
                 neural=True, anchor=True, diag=False, state_source="chord"):
        self.g, self.agg, self.anchor, self.fuse = grid, agg, anchor, fuse
        self.neural, self.diag = neural, diag
        t0 = time.time()
        self.se = None
        if state_source == "chord":
            self.se = ChordSE(grid, iters=se_iters)
            self.est = lambda Z: self.se.estimate(Z)
        elif state_source == "wls":  # full iterative Gauss-Newton WLS (flat start)
            self.est = lambda Z: grid.wls(Z)[:2]
        elif state_source == "learned":  # learned MLP inverse trained with the AC physical loss (PI-BAE-style encoder)
            enc = A.PIAE(grid.m, grid.n, A.WIDTHS[ds.base], A.PhysLayer(grid, ds.mu, ds.sd), w_phy=1.0)
            enc, _ = A.train(enc, ds.norm(ds.normals("train")), ds.norm(ds.normals("val")))

            @torch.no_grad()
            def _est(Z):
                zn = torch.tensor(ds.norm(Z), device=DEV)
                X = torch.cat([enc.P.to_state(enc.E(zn[i:i + 4096])) for i in range(0, len(zn), 4096)])
                return X.double().cpu().numpy(), np.zeros(len(Z))
            self.est = _est
            self.enc = enc
        Ztr = ds.normals("train")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Ztr))
        ncal = int(cal_frac * len(Ztr)) if (fuse or agg == "maxcal") else 0
        cal, fit = perm[:ncal], perm[ncal:]
        Xtr, _ = self.est(Ztr); Xv, _ = self.est(ds.normals("val"))
        self.rep = StateRep(grid, Xtr[fit])
        Str = torch.tensor(self.rep(Xtr[fit]), device=DEV); Sv = torch.tensor(self.rep(Xv), device=DEV)
        self.dimmask = torch.tensor(self.rep.dimmask, device=DEV)
        self._fit_anchor(Str)
        if not anchor:  # ablation: no linear LOO anchor (network must model s_b from u_b alone)
            self.a_P = torch.zeros_like(self.a_P); self.a_mu = torch.zeros_like(self.a_mu)
        # whitening of the anchored residual by its linear conditional covariance (Lambda_bb^{-1})
        Binv = self.a_Binv.float()
        if whiten and anchor:
            Lc = torch.linalg.cholesky(Binv + 1e-9 * torch.eye(2, device=DEV) + torch.diag_embed(1 - self.dimmask))
        else:
            Lc = torch.eye(2, device=DEV).expand(grid.N, 2, 2).clone()
        self.Lc = Lc
        self.Lc_inv = torch.linalg.inv(Lc)
        self.se_time = time.time() - t0
        torch.manual_seed(seed)
        self.model = BlindSpotNet(grid.N, r, hidden, emb, depth).to(DEV)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, bad, bst = np.inf, 0, copy.deepcopy(self.model.state_dict())
        t1 = time.time()
        for ep in range(epochs if neural else 0):
            self.model.train()
            p = torch.randperm(len(Str), device=DEV)
            for i in range(0, len(p), bs):
                loss = self._loss(Str[p[i:i + bs]])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
            sched.step()
            self.model.eval()
            with torch.no_grad():
                vl = self._loss(Sv).item()
            if vl < best - 1e-5:
                best, bad, bst = vl, 0, copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        self.model.load_state_dict(bst); self.model.eval()
        self.train_time, self.epochs = time.time() - t1, (ep + 1 if neural else 0)
        self.infer_masks = torch.zeros(1, grid.N, dtype=torch.bool, device=DEV)  # single pass
        self._post_fit(Xtr[cal], fuse)

    def _d2(self, s):
        e = self.target(s)                                           # linear LOO residual  [B,N,2]
        w = torch.einsum("nij,bnj->bni", self.Lc_inv, e) * self.dimmask  # whitened by linear conditional cov
        lin = (s - e) * self.dimmask                                 # linear LOO prediction of s_b
        if not self.anchor:
            lin = torch.zeros_like(lin)
        mu, l11, l22, l21 = self.model(s * self.dimmask, lin)
        if self.diag:
            l21 = torch.zeros_like(l21)
        return gauss2_d2(w, mu, l11, l22, l21, self.dimmask)

    def _loss(self, s):
        d2, ld = self._d2(s)
        return (0.5 * d2 + ld).mean()

    @torch.no_grad()
    def bus_stats_from_states(self, X, bs=4096):
        S = torch.tensor(self.rep(X), device=DEV)
        return torch.cat([self._d2(S[i:i + bs])[0] for i in range(0, len(S), bs)]).cpu().numpy()


@register("BSP")
def _bsp(ds, grid, seed=0, **kw):
    return BSP(ds, grid, seed, **kw)


# ============================================================================ frozen final model + ablations
def _final_hp():
    import json as _json
    import os as _os
    from common import ROOT as _ROOT
    p = _os.path.join(_ROOT, "configs", "final_hparams.json")
    return _json.load(open(p))


ABLATIONS = {  # name -> overrides of the frozen configuration (one component removed / replaced at a time)
    "PALOC": {},
    "PALOC-noNN": {"neural": False},          # linear-Gaussian LOO anchor only
    "PALOC-noAnchor": {"anchor": False},      # blind-spot network without the linear anchor
    "PALOC-diag": {"diag": True},             # diagonal instead of full 2x2 covariance
    "PALOC-sum": {"agg": "sum"},              # global pooling (sum over buses) instead of localised pooling
    "PALOC-learnedSE": {"state_source": "learned"},  # learned MLP inverse (PI-AE encoder) instead of physics SE
    "PALOC-fullWLS": {"state_source": "wls"},  # full iterative WLS instead of fixed-gain chord SE
}


def _make_final(name):
    def _b(ds, grid, seed=0, **kw):
        p = {**_final_hp(), **ABLATIONS[name], **kw}
        return BSP(ds, grid, seed, **p)
    return _b


for _n in ABLATIONS:
    register(_n)(_make_final(_n))
