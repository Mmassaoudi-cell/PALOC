"""Autoencoder-family detectors, incl. the faithful reproduction of the source method (PI-BAE)
and the source paper's baselines (SAE, VAE, GraphAE, PI-AE), plus Deep SVDD.

Training protocol (source Table I): Adam lr=1e-3, betas=(0.9,0.999), wd=1e-5, batch 128,
200 epochs, early stopping patience 20 on validation-NORMAL loss, grad-clip 1.0.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from common import DEV

WIDTHS = {  # encoder hidden widths (source Table I); 39-bus is our assumption (A10)
    "ieee14": [256, 128, 64],
    "ieee39": [256, 128],
    "ieee118": [512, 256],
}


def mlp(sizes, bias=True, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.LeakyReLU(0.01))
    return nn.Sequential(*layers)


class PhysLayer(nn.Module):
    """Non-trainable differentiable AC measurement decoder: u -> x_hat -> standardised h(x_hat)."""

    def __init__(self, grid, mu, sd):
        super().__init__()
        self.grid = grid
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.N = grid.N

    def to_state(self, u):  # data-free affine prior around flat start (assumption A6)
        return torch.cat([1.0 + 0.1 * u[:, : self.N], 0.2 * u[:, self.N:]], 1)

    def forward(self, u):
        z = self.grid.h(self.to_state(u))
        return (z - self.mu) / self.sd


class Base(nn.Module):
    def loss(self, z):
        raise NotImplementedError

    def score(self, z):
        raise NotImplementedError


class SAE(Base):
    def __init__(self, m, n, widths):
        super().__init__()
        self.E = mlp([m] + widths + [n])
        self.D = mlp([n] + widths[::-1] + [m])

    def loss(self, z):
        return ((z - self.D(self.E(z))) ** 2).mean()

    def score(self, z):
        return ((z - self.D(self.E(z))) ** 2).mean(1)


class VAE(Base):
    def __init__(self, m, n, widths):
        super().__init__()
        self.E = mlp([m] + widths + [2 * n])
        self.D = mlp([n] + widths[::-1] + [m])
        self.n = n

    def loss(self, z):
        h = self.E(z); mu, lv = h[:, : self.n], h[:, self.n:].clamp(-10, 10)
        zz = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        rec = 0.5 * ((z - self.D(zz)) ** 2).sum(1)
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
        return (rec + kl).mean() / z.shape[1]

    def score(self, z):
        mu = self.E(z)[:, : self.n]
        return ((z - self.D(mu)) ** 2).mean(1)


class PIAE(Base):
    """Physics-informed AE with fixed weights (w_phy=1) -- source baseline."""

    def __init__(self, m, n, widths, phys, w_phy=1.0):
        super().__init__()
        self.E = mlp([m] + widths + [n])
        self.D = mlp([n] + widths[::-1] + [m])
        self.P = phys
        self.w = w_phy

    def residuals(self, z):
        u = self.E(z)
        return ((z - self.D(u)) ** 2).mean(1), ((z - self.P(u)) ** 2).mean(1)

    def loss(self, z):
        rs, rp = self.residuals(z)
        return (rs + self.w * rp).mean()

    def score(self, z):
        rs, rp = self.residuals(z)
        return rs + self.w * rp


class PIBAE(PIAE):
    """SOURCE_METHOD_REPRODUCTION: PI-BAE (Eqs. 8-15 of the source paper)."""

    def __init__(self, m, n, widths, phys):
        super().__init__(m, n, widths, phys)
        self.log_s1 = nn.Parameter(torch.zeros(()))
        self.log_s2 = nn.Parameter(torch.zeros(()))

    def loss(self, z):  # Eq. (13)
        rs, rp = self.residuals(z)
        return (rs.mean() / (2 * torch.exp(2 * self.log_s1)) + rp.mean() / (2 * torch.exp(2 * self.log_s2))
                + self.log_s1 + self.log_s2)

    def score(self, z):  # Eq. (14)
        rs, rp = self.residuals(z)
        return rs / torch.exp(2 * self.log_s1) + rp / torch.exp(2 * self.log_s2)


class GraphAE(Base):
    """Topology-aware graph autoencoder (source baseline; architecture assumed, A8)."""

    def __init__(self, grid, m, hidden=64, lat=2):
        super().__init__()
        N = grid.N
        A = grid.adjacency().astype(np.float32) + np.eye(N, dtype=np.float32)
        dinv = 1 / np.sqrt(A.sum(1))
        self.register_buffer("A", torch.tensor(A * dinv[:, None] * dinv[None, :]))
        # node feature gather: V, P, Q, sum of from-end P and Q flows (channel index maps)
        M = np.zeros((5, N, m), np.float32)
        for c, (tp, b) in enumerate(zip(grid.ch_type, grid.ch_bus)):
            k = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 4}[tp]
            M[k, b, c] = 1.0
        self.register_buffer("M", torch.tensor(M))
        self.W1, self.W2 = nn.Linear(5, hidden), nn.Linear(hidden, lat)
        self.D = mlp([lat * N, 256, m])
        self.act = nn.LeakyReLU(0.01)

    def recon(self, z):
        X = torch.einsum("knm,bm->bnk", self.M, z)
        H = self.act(self.A @ self.W1(X))
        H = self.A @ self.W2(H)
        return self.D(H.flatten(1))

    def loss(self, z):
        return ((z - self.recon(z)) ** 2).mean()

    def score(self, z):
        return ((z - self.recon(z)) ** 2).mean(1)


class DeepSVDD(Base):
    def __init__(self, m, widths, rep=32):
        super().__init__()
        self.E = mlp([m] + widths + [rep], bias=False)
        self.register_buffer("c", torch.zeros(rep))
        self.init = False

    def loss(self, z):
        if not self.init:
            with torch.no_grad():
                c = self.E(z).mean(0)
                c[(c.abs() < 0.01)] = 0.01
                self.c.copy_(c)
            self.init = True
        return ((self.E(z) - self.c) ** 2).sum(1).mean()

    def score(self, z):
        return ((self.E(z) - self.c) ** 2).sum(1)


def train(model, Ztr, Zval, epochs=200, bs=128, lr=1e-3, wd=1e-5, patience=20, clip=1.0, verbose=False):
    """Generic trainer with early stopping on validation-normal loss (no attack labels)."""
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    Ztr_t = torch.tensor(Ztr, device=DEV); Zv = torch.tensor(Zval, device=DEV)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Ztr_t), device=DEV)
        for i in range(0, len(perm), bs):
            loss = model.loss(Ztr_t[perm[i:i + bs]])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([model.loss(Zv[i:i + 1024]).item() for i in range(0, len(Zv), 1024)]))
        hist.append(vl)
        if vl < best - 1e-7:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(epochs=ep + 1, train_time=time.time() - t0, best_val_loss=best, hist=hist)


@torch.no_grad()
def score_np(model, Zn, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(Zn), bs):
        out.append(model.score(torch.tensor(Zn[i:i + bs], device=DEV)).float().cpu().numpy())
    return np.concatenate(out)
