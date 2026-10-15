"""Fixed-gain (chord) Gauss-Newton AC state estimator.

x_{k+1} = x_k + G0 (z - h(x_k)),   G0 = (H0^T W H0)^{-1} H0^T W,  H0 = H(x_nom)
G0 is computed once offline at the nominal operating point (the defender's base case), so each
iteration costs one evaluation of h and one dense n x m matrix-vector product: no Jacobian
evaluation or linear solve online.
"""
import numpy as np
import torch

from common import DEV


class ChordSE:
    def __init__(self, grid, iters=3, device=DEV, dtype=torch.float64):
        self.g, self.iters, self.dev, self.dt = grid, iters, device, dtype
        x_nom = grid.state_from_net(grid.net)
        self.x0 = torch.tensor(x_nom, device=device, dtype=dtype)
        H0 = grid.jac(self.x0[None].double())[0].double()
        W = torch.tensor(1.0 / grid.sigma ** 2, device=device, dtype=torch.float64)
        HtW = H0.T * W
        self.H0 = H0
        self.G0 = torch.linalg.solve(HtW @ H0, HtW).to(dtype)  # [n, m]
        self.W = W.to(dtype)

    def _masked_gain(self, obs):
        """Per-snapshot gain with zero weight on missing channels: G = (H0^T W_m H0)^-1 H0^T W_m, obs [B, m] bool."""
        Wm = self.W.double() * obs.double()                       # [B, m]
        HtW = self.H0.T.unsqueeze(0) * Wm.unsqueeze(1)            # [B, n, m]
        A = HtW @ self.H0                                          # [B, n, n]
        A = A + 1e-9 * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        return torch.linalg.solve(A, HtW).to(self.dt)             # [B, n, m]

    @torch.no_grad()
    def estimate(self, Z, iters=None, chunk=8192, return_J=True):
        """Missing channels may be given as NaN: they receive zero weight (native missing-data handling)."""
        iters = self.iters if iters is None else iters
        xs, Js = [], []
        Z = np.asarray(Z)
        for s in range(0, len(Z), chunk):
            z = torch.tensor(Z[s:s + chunk], device=self.dev, dtype=self.dt)
            obs = torch.isfinite(z)
            x = self.x0.expand(len(z), -1).clone()
            if bool(obs.all()):
                for _ in range(iters):
                    x = x + (z - self.g.h(x)) @ self.G0.T
                r = z - self.g.h(x)
            else:
                z0 = torch.where(obs, z, torch.zeros_like(z))
                parts_x, parts_r = [], []
                for i in range(0, len(z), 256):
                    zi, oi, xi = z0[i:i + 256], obs[i:i + 256], x[i:i + 256]
                    G = self._masked_gain(oi)
                    for _ in range(iters):
                        res = torch.where(oi, zi - self.g.h(xi), torch.zeros_like(zi))
                        xi = xi + torch.einsum("bnm,bm->bn", G, res)
                    parts_x.append(xi)
                    parts_r.append(torch.where(oi, zi - self.g.h(xi), torch.zeros_like(zi)))
                x, r = torch.cat(parts_x), torch.cat(parts_r)
            xs.append(x.cpu().numpy()); Js.append((r ** 2 * self.W).sum(1).cpu().numpy())
        X = np.concatenate(xs)
        return (X, np.concatenate(Js)) if return_J else X

    def macs(self):
        g = self.g
        h_macs = 4 * g.N * g.N + 4 * g.L * g.N * 2  # dense implementation (upper bound)
        return self.iters * (g.n * g.m + h_macs) + h_macs
