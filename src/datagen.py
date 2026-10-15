"""Dataset generation: normal operating points (pandapower AC power flow), noisy measurements,
and AC-stealthy FDIAs (Eq. a = h(x+c) - h(x), retained if WLS-BDD is evaded).

Each (system, replicate) is an independent dataset: new load scenarios, noise, splits and attacks.
Usage: python datagen.py --system ieee14 --reps 0 1 2 3 4
"""
import argparse
import copy
import os
import time
from multiprocessing import Pool

import numpy as np
import pandapower as pp
import torch
from pandapower.pypower.makeYbus import makeYbus

from grid import Grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
N_NORMAL, N_TRAIN, N_VAL = 10000, 7000, 1500
V_LIM_NORMAL = (0.90, 1.10)
V_LIM_ATTACK = (0.88, 1.12)
TIERS = {  # |dV| range (p.u.), |dtheta| range (deg)
    "S": (0.02, 0.05, 2.0, 6.0),     # source-paper setting
    "M": (0.004, 0.01, 0.4, 1.2),    # 0.2x
    "W": (0.001, 0.0025, 0.1, 0.3),  # 0.05x (weak FDIA)
}
MAX_CH = 30

_G = {}


def _init(system):
    g = Grid(system)
    _G["g"] = g
    _G["p0"] = g.net.load.p_mw.values.copy()
    _G["q0"] = g.net.load.q_mvar.values.copy()


def _solve(factors):
    g = _G["g"]; net = g.net
    out = []
    for fac in factors:
        net.load["p_mw"] = _G["p0"] * fac
        net.load["q_mvar"] = _G["q0"] * fac
        try:
            pp.runpp(net, init="results", max_iteration=20)
            ok = net.converged
        except Exception:
            ok = False
        if not ok:
            out.append(None); continue
        x = g.state_from_net(net)
        vm = x[: g.N]
        out.append(x if (vm.min() >= V_LIM_NORMAL[0] and vm.max() <= V_LIM_NORMAL[1]) else None)
    return out


def sample_states(system, n, rng, load_range=(0.8, 1.2), procs=16):
    g = Grid(system)
    nl = len(g.net.load)
    X = []
    tries = 0
    with Pool(procs, initializer=_init, initargs=(system,)) as pool:
        while len(X) < n:
            k = int((n - len(X)) * 1.1) + 64
            F = rng.uniform(*load_range, size=(k, nl))
            chunks = np.array_split(F, procs * 4)
            for res in pool.map(_solve, chunks):
                X += [r for r in res if r is not None]
            tries += k
    return np.array(X[:n]), tries


def eligible_buses(g, x_nom):
    el = []
    for b in g.nonref:
        if g.affected_channels(b, x_nom).sum() <= MAX_CH:
            el.append(int(b))
    return np.array(el)


def make_c(g, buses, tiers, rng):
    """Attack state perturbation vectors c [K,n]."""
    K = len(buses)
    c = np.zeros((K, g.n))
    dv = np.zeros(K); dth = np.zeros(K)
    for k, (b, t) in enumerate(zip(buses, tiers)):
        v0, v1, a0, a1 = TIERS[t]
        dv[k] = rng.uniform(v0, v1) * rng.choice([-1, 1])
        dth[k] = np.deg2rad(rng.uniform(a0, a1)) * rng.choice([-1, 1])
        c[k, b] = dv[k]
        c[k, g.N + int(np.where(g.nonref == b)[0][0])] = dth[k]
    return c, dv, dth


def h_np(g, X, chunk=4096):
    out = []
    for s in range(0, len(X), chunk):
        out.append(g.h(torch.tensor(X[s:s + chunk], dtype=torch.float64, device="cuda")).cpu().numpy())
    return np.concatenate(out)


def perturbed_grid(g, rng, spread=0.1):
    """Attacker's imperfect model: branch r, x, b each scaled by U[1-spread, 1+spread]."""
    g2 = copy.copy(g)
    ppc = g.net._ppc
    branch = ppc["branch"].copy()
    L = branch.shape[0]
    for col in (2, 3, 4):
        branch[:, col] = branch[:, col] * rng.uniform(1 - spread, 1 + spread, L)
    Y, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], branch)
    g2.Ybus, g2.Yf, g2.Yt = Y.toarray(), Yf.toarray(), Yt.toarray()
    g2._torch_cache = {}
    return g2


def build_attacks(g, X, E, base_idx, tiers, eligible, rng, kind="stealthy", g_att=None, max_rounds=8):
    """Generate one retained attack per entry of base_idx. Returns dict of arrays."""
    K = len(base_idx)
    done = np.zeros(K, bool)
    res = dict(Z=np.zeros((K, g.m), np.float32), bus=np.zeros(K, int), dv=np.zeros(K), dth=np.zeros(K),
               J=np.zeros(K), nch=np.zeros(K, int), xa=np.zeros((K, g.n)))
    tiers = np.asarray(tiers)
    for rnd in range(max_rounds):
        todo = np.where(~done)[0]
        if len(todo) == 0:
            break
        buses = rng.choice(eligible, size=len(todo))
        c, dv, dth = make_c(g, buses, tiers[todo], rng)
        Xb = X[base_idx[todo]]
        if kind == "gray":
            a = h_np(g_att, Xb + c) - h_np(g_att, Xb)
        else:
            a = h_np(g, Xb + c) - h_np(g, Xb)
        a[np.abs(a) < 1e-10] = 0.0
        if kind == "generic":  # same support and norm, random direction (not AC-consistent)
            r = rng.normal(size=a.shape) * (a != 0)
            r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12) * np.linalg.norm(a, axis=1, keepdims=True)
            a = r
        En = E[base_idx[todo]] if rnd == 0 else rng.normal(size=a.shape) * g.sigma
        Za = h_np(g, Xb) + En + a
        xh, J, conv = g.wls(Za)
        vm = (Xb + c)[:, : g.N]
        lim = (vm.min(1) >= V_LIM_ATTACK[0]) & (vm.max(1) <= V_LIM_ATTACK[1])
        nch = (a != 0).sum(1)
        if kind == "stealthy":
            ok = conv & (J < g.tau_bdd) & lim & (nch <= MAX_CH)
        else:  # generic / gray-box attacks are not filtered by BDD (defender-side ground truth)
            ok = lim & (nch <= MAX_CH)
        acc = todo[ok]
        res["Z"][acc] = Za[ok]; res["bus"][acc] = buses[ok]; res["dv"][acc] = dv[ok]; res["dth"][acc] = dth[ok]
        res["J"][acc] = J[ok]; res["nch"][acc] = nch[ok]; res["xa"][acc] = (Xb + c)[ok]
        done[acc] = True
    keep = np.where(done)[0]
    out = {k: v[keep] for k, v in res.items()}
    out["base"] = base_idx[keep]
    out["tier"] = tiers[keep]
    return out


def generate(system, rep):
    t0 = time.time()
    seed = {"ieee14": 1000, "ieee39": 2000, "ieee118": 3000}[system] + rep
    rng = np.random.default_rng(seed)
    g = Grid(system)
    X, tries = sample_states(system, N_NORMAL, rng)
    perm = rng.permutation(N_NORMAL)
    split = np.empty(N_NORMAL, dtype="<U5")
    split[perm[:N_TRAIN]] = "train"; split[perm[N_TRAIN:N_TRAIN + N_VAL]] = "val"; split[perm[N_TRAIN + N_VAL:]] = "test"
    E = rng.normal(size=(N_NORMAL, g.m)) * g.sigma
    Z = h_np(g, X) + E
    xh, J, conv = g.wls(Z)
    x_nom = g.state_from_net(g.net)
    elig = eligible_buses(g, x_nom)
    out = dict(X=X, Z=Z.astype(np.float32), split=split, J=J, wls_conv=conv, eligible=elig, tries=tries)
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    att = {}
    # independent noise draws for attacked versions (attack built on same operating point, new noise)
    E2 = rng.normal(size=(N_NORMAL, g.m)) * g.sigma
    for t in TIERS:
        att[f"test_{t}"] = build_attacks(g, X, E2, idx["test"], [t] * len(idx["test"]), elig, rng)
    rr = lambda n: np.array(list(TIERS))[np.arange(n) % 3]
    att["val_mix"] = build_attacks(g, X, E2, idx["val"], rr(len(idx["val"])), elig, rng)
    att["train_mix"] = build_attacks(g, X, E2, idx["train"], rr(len(idx["train"])), elig, rng)
    att["test_generic"] = build_attacks(g, X, E2, idx["test"], ["S"] * len(idx["test"]), elig, rng, kind="generic")
    g_att = perturbed_grid(g, rng)
    att["test_gray"] = build_attacks(g, X, E2, idx["test"], ["S"] * len(idx["test"]), elig, rng, kind="gray", g_att=g_att)
    for k, v in att.items():
        for kk, vv in v.items():
            out[f"A_{k}_{kk}"] = vv
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(DATA_DIR, f"{system}_rep{rep}.npz"), **out)
    print(f"{system} rep{rep}: tries={tries} elig={len(elig)} J_normal_mean={J.mean():.1f} "
          f"BDD_FPR_normal={(J > g.tau_bdd).mean():.3f} " +
          " ".join(f"{k}:{len(v['bus'])}" for k, v in att.items()) + f"  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--reps", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    for r in a.reps:
        generate(a.system, r)
