"""Power-grid substrate: pandapower cases, differentiable AC measurement function h(x),
batched Gauss-Newton WLS state estimation, and chi-square bad-data detection.

Measurement ordering (m = N + 2(N-1) + 4L):
    [Vm (N) | P_inj non-ref (N-1) | Q_inj non-ref (N-1) | Pf (L) | Qf (L) | Pt (L) | Qt (L)]
State ordering (n = 2N-1):
    [Vm (N) | theta non-ref (N-1)]   (radians, reference angle fixed at 0)
"""
import re
import warnings

import numpy as np
import torch
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from scipy.stats import chi2

warnings.filterwarnings("ignore")

SIGMA_V, SIGMA_PQ = 0.002, 0.01
CASES = {"ieee14": "case14", "ieee39": "case39", "ieee118": "case118"}


class Grid:
    def __init__(self, system: str):
        self.system = system
        net = getattr(pn, CASES[re.match(r"(ieee\d+)", system).group(1)])()
        pp.runpp(net)
        self.net = net
        ppc = net._ppc
        self.baseMVA = ppc["baseMVA"]
        bus, branch = ppc["bus"], ppc["branch"]
        Ybus, Yf, Yt = makeYbus(self.baseMVA, bus, branch)
        self.Ybus, self.Yf, self.Yt = Ybus.toarray(), Yf.toarray(), Yt.toarray()
        self.N, self.L = bus.shape[0], branch.shape[0]
        self.ref = int(np.where(bus[:, 1] == 3)[0][0])
        self.nonref = np.array([i for i in range(self.N) if i != self.ref])
        self.f = branch[:, 0].real.astype(int)
        self.t = branch[:, 1].real.astype(int)
        self.n = 2 * self.N - 1
        self.m = self.N + 2 * (self.N - 1) + 4 * self.L
        self.bus_lookup = net._pd2ppc_lookups["bus"]
        # channel metadata: type and associated bus (for localisation / graph features)
        N, L, nr = self.N, self.L, self.nonref
        self.ch_type = np.concatenate([np.zeros(N), np.ones(N - 1), 2 * np.ones(N - 1),
                                       3 * np.ones(L), 4 * np.ones(L), 5 * np.ones(L), 6 * np.ones(L)]).astype(int)
        self.ch_bus = np.concatenate([np.arange(N), nr, nr, self.f, self.f, self.t, self.t]).astype(int)
        self.ch_branch = np.concatenate([-np.ones(N + 2 * (N - 1)), np.tile(np.arange(L), 4)]).astype(int)
        self.sigma = np.where(self.ch_type == 0, SIGMA_V, SIGMA_PQ)
        self.state_bus = np.concatenate([np.arange(N), nr]).astype(int)
        self.tau_bdd = chi2.ppf(0.95, self.m - self.n)
        self._torch_cache = {}

    # ------------------------------------------------------------------ torch tensors
    def T(self, device="cpu", dtype=torch.float64):
        key = (str(device), dtype)
        if key not in self._torch_cache:
            mk = lambda a: torch.tensor(a, device=device, dtype=dtype)
            self._torch_cache[key] = dict(
                G=mk(self.Ybus.real), B=mk(self.Ybus.imag),
                Gf=mk(self.Yf.real), Bf=mk(self.Yf.imag),
                Gt=mk(self.Yt.real), Bt=mk(self.Yt.imag),
                f=torch.tensor(self.f, device=device), t=torch.tensor(self.t, device=device),
                nonref=torch.tensor(self.nonref, device=device),
            )
        return self._torch_cache[key]

    def split_state(self, x):
        """x [..., n] -> Vm [..., N], theta [..., N] with reference angle 0."""
        Vm = x[..., : self.N]
        th = torch.zeros_like(Vm)
        th = th.index_copy(-1, self.T(x.device, x.dtype)["nonref"], x[..., self.N:])
        return Vm, th

    def h(self, x):
        """Differentiable AC measurement function, batched over leading dims. x [..., n] -> [..., m]."""
        P = self.T(x.device, x.dtype)
        Vm, th = self.split_state(x)
        vr, vi = Vm * torch.cos(th), Vm * torch.sin(th)
        ir = vr @ P["G"].T - vi @ P["B"].T
        ii = vi @ P["G"].T + vr @ P["B"].T
        Pinj = vr * ir + vi * ii
        Qinj = vi * ir - vr * ii
        ifr = vr @ P["Gf"].T - vi @ P["Bf"].T
        ifi = vi @ P["Gf"].T + vr @ P["Bf"].T
        itr = vr @ P["Gt"].T - vi @ P["Bt"].T
        iti = vi @ P["Gt"].T + vr @ P["Bt"].T
        vfr, vfi = vr[..., P["f"]], vi[..., P["f"]]
        vtr, vti = vr[..., P["t"]], vi[..., P["t"]]
        Pf = vfr * ifr + vfi * ifi
        Qf = vfi * ifr - vfr * ifi
        Pt = vtr * itr + vti * iti
        Qt = vti * itr - vtr * iti
        nr = P["nonref"]
        return torch.cat([Vm, Pinj[..., nr], Qinj[..., nr], Pf, Qf, Pt, Qt], dim=-1)

    def jac(self, x):
        """Batched Jacobian H(x) [B, m, n] via forward-mode AD."""
        return torch.func.vmap(torch.func.jacfwd(self.h))(x)

    def flat_state(self, B, device="cpu", dtype=torch.float64):
        x = torch.zeros(B, self.n, device=device, dtype=dtype)
        x[:, : self.N] = 1.0
        return x

    # ------------------------------------------------------------------ WLS
    @torch.no_grad()
    def wls(self, z, x0=None, max_iter=30, tol=1e-6, chunk=256, device="cuda"):
        """Batched Gauss-Newton WLS. Returns x_hat [B,n], J [B], converged [B] (numpy)."""
        z = torch.as_tensor(z, dtype=torch.float64)
        W = torch.tensor(1.0 / self.sigma ** 2, device=device)
        xs, Js, cs = [], [], []
        for s in range(0, z.shape[0], chunk):
            zc = z[s:s + chunk].to(device)
            x = self.flat_state(zc.shape[0], device) if x0 is None else torch.as_tensor(x0[s:s + chunk], dtype=torch.float64, device=device)
            conv = torch.zeros(zc.shape[0], dtype=torch.bool, device=device)
            for _ in range(max_iter):
                r = zc - self.h(x)
                H = self.jac(x)
                HtW = H.transpose(1, 2) * W
                dx = torch.linalg.solve(HtW @ H, (HtW @ r.unsqueeze(-1))).squeeze(-1)
                dx = torch.where(conv[:, None], torch.zeros_like(dx), dx)
                x = x + dx
                conv = conv | (dx.abs().amax(1) < tol)
                if bool(conv.all()):
                    break
            r = zc - self.h(x)
            J = (r ** 2 * W).sum(1)
            ok = conv & torch.isfinite(J)
            xs.append(x.cpu()); Js.append(J.cpu()); cs.append(ok.cpu())
        return torch.cat(xs).numpy(), torch.cat(Js).numpy(), torch.cat(cs).numpy()

    # ------------------------------------------------------------------ state from pandapower
    def state_from_net(self, net):
        lk = self.bus_lookup[net.bus.index.values]
        vm = np.zeros(self.N); va = np.zeros(self.N)
        vm[lk] = net.res_bus.vm_pu.values
        va[lk] = np.deg2rad(net.res_bus.va_degree.values)
        va = va - va[self.ref]
        return np.concatenate([vm, va[self.nonref]])

    def adjacency(self):
        A = np.zeros((self.N, self.N), dtype=bool)
        A[self.f, self.t] = True
        A[self.t, self.f] = True
        np.fill_diagonal(A, False)
        return A

    def affected_channels(self, bus, x_nom, dv=0.03, dth=np.deg2rad(4)):
        x = torch.tensor(x_nom[None], dtype=torch.float64)
        c = torch.zeros_like(x)
        c[0, bus] = dv
        if bus != self.ref:
            c[0, self.N + int(np.where(self.nonref == bus)[0][0])] = dth
        a = (self.h(x + c) - self.h(x))[0].numpy()
        return np.abs(a) > 1e-10
