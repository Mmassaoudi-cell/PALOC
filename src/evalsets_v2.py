"""Extended evaluation sets (v2), generated deterministically per (system, replicate) and cached.

Adversarial (detector-aware) AC-stealthy attacks, built for VALIDATION (model selection) and TEST:
  A_alig{M,W}   single-bus shift along the direction of LARGEST conditional variance of the bus state given the
                rest of the grid (white-box attacker who knows the defender's normal-operation statistics);
                magnitude matched to the tier (same normalised norm as a random tier attack)
  A_multiM      coordinated attack on 2-3 adjacent buses, random directions, tier-M energy split over the buses
  A_multialigM  as A_multiM but every bus shifted along its largest-conditional-variance direction
  A_lr          load-redistribution attack: the reported state is the exact AC operating point obtained when
                +dP is moved from one load bus to another (dP = 20-50% of the larger bus load, Q at the bus
                power factor), i.e. a physically valid state that mimics a load change
Operating-condition sets (TEST, with VALIDATION-state recalibration pools *_V):
  noise{k}_V           validation states, noise x k (k = 0.5, 1.5, 2)    -> threshold recalibration
  missnan{q}_{N,A_M,A_W,V}  fraction q of channels missing, encoded as NaN (q = 0.05, 0.10)
  topo_{N,A_M,V}       N-1 line outage (3 fixed non-bridge lines, cycled); states re-solved by AC power flow with
                       the injections of the intact snapshot; attacks built with the outaged network model
All attacks are retained only if flat-start WLS (with the network model in force) converges and J < chi2_0.95.
"""
import copy
import os

import networkx as nx
import numpy as np
import torch
from pandapower.pypower.makeYbus import makeYbus
from sklearn.covariance import LedoitWolf

import datagen as D
from common import DATA_DIR
from physics import ChordSE

NOISE = [0.5, 1.5, 2.0]
MISS = [0.05, 0.10]
TIER_MID = {t: ((v0 + v1) / 2, np.deg2rad((a0 + a1) / 2)) for t, (v0, v1, a0, a1) in D.TIERS.items()}


def _h(g, X, dev="cuda"):
    out = []
    for i in range(0, len(X), 4096):
        out.append(g.h(torch.tensor(X[i:i + 4096], dtype=torch.float64, device=dev)).cpu().numpy())
    return np.concatenate(out)


def bus_types(g):
    bt = g.net._ppc["bus"][:, 1].real.astype(int)
    return bt  # 1 PQ, 2 PV, 3 ref


def outage_grid(g, line):
    g2 = copy.copy(g)
    ppc = g.net._ppc
    br = ppc["branch"].copy()
    br[line, 10] = 0  # BR_STATUS
    Y, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], br)
    g2.Ybus, g2.Yf, g2.Yt = Y.toarray(), Yf.toarray(), Yt.toarray()
    g2._torch_cache = {}
    return g2


def pick_outage_lines(g, k=3, seed=0):
    G = nx.MultiGraph()
    G.add_nodes_from(range(g.N))
    for i, (f, t) in enumerate(zip(g.f, g.t)):
        G.add_edge(f, t, key=i)
    bridges = set()
    simple = nx.Graph(G)
    for u, v in nx.bridges(simple):
        if G.number_of_edges(u, v) == 1:
            bridges.add(frozenset((u, v)))
    cand = [i for i, (f, t) in enumerate(zip(g.f, g.t)) if frozenset((f, t)) not in bridges and f != t]
    # most loaded candidate lines in the base case (largest |Pf|)
    x0 = g.state_from_net(g.net)
    z0 = g.h(torch.tensor(x0[None])).numpy()[0]
    pf = np.abs(z0[g.N + 2 * (g.N - 1): g.N + 2 * (g.N - 1) + g.L])
    cand = sorted(cand, key=lambda i: -pf[i])[: max(k * 3, k)]
    rng = np.random.default_rng(seed)
    return list(rng.choice(cand, k, replace=False))


def pf_solve(g, x0, Pt, Qt, Vt, pq_mask, iters=12, dev="cuda"):
    """Batched AC power flow with our h: unknown full state x; equations
    P_nonref(x)=Pt, Q_i(x)=Qt for PQ buses, V_i = Vt for PV and reference buses (n equations)."""
    # unknowns: all V (N) + theta non-ref (N-1) = n; equations: P (N-1) + [Q at PQ | V at PV] (N-1) + V_ref (1) = n
    N = g.N
    pqm = torch.tensor(pq_mask, device=dev)
    xs_all, ok_all = [], []
    for s in range(0, len(x0), 256):
        x = torch.tensor(x0[s:s + 256], dtype=torch.float64, device=dev)
        a = torch.tensor(Pt[s:s + 256], dtype=torch.float64, device=dev)
        b = torch.tensor(Qt[s:s + 256], dtype=torch.float64, device=dev)
        c = torch.tensor(Vt[s:s + 256], dtype=torch.float64, device=dev)
        fun = lambda xx, aa, bb, cc: _F1(g, xx, aa, bb, cc, pqm, N)
        for _ in range(iters):
            f = torch.func.vmap(fun)(x, a, b, c)
            Jm = torch.func.vmap(torch.func.jacfwd(fun))(x, a, b, c)
            dx = torch.linalg.lstsq(Jm, -f.unsqueeze(-1)).solution.squeeze(-1)
            x = x + dx
            if float(dx.abs().max()) < 1e-10:
                break
        f = torch.func.vmap(fun)(x, a, b, c)
        xs_all.append(x.cpu().numpy())
        ok_all.append(((f.abs().amax(1) < 1e-6) & torch.isfinite(f).all(1)).cpu().numpy())
    return np.concatenate(xs_all), np.concatenate(ok_all)


def _F1(g, xx, Pt, Qt, Vt, pqm, N):
    z = g.h(xx)
    P = z[N:2 * N - 1]; Q = z[2 * N - 1:3 * N - 2]
    nr = g.T(xx.device, xx.dtype)["nonref"]
    fQ = torch.where(pqm, Q - Qt, xx[:N][nr] - Vt[nr])
    return torch.cat([P - Pt, fQ, (xx[:N] - Vt)[[g.ref]]])


def specs_from_state(g, X):
    """Injection / voltage specification of the operating points (for re-solving power flows)."""
    z = _h(g, X)
    N = g.N
    P = z[:, N:2 * N - 1]; Q = z[:, 2 * N - 1:3 * N - 2]; V = X[:, :N]
    pq = bus_types(g)[g.nonref] == 1
    return P, Q, V, pq


def anchor_directions(g, se, Ztr):
    """Largest-conditional-variance direction of each bus state (physical units, unit 'tier-normalised' norm)."""
    Xtr, _ = se.estimate(Ztr)
    N = g.N
    V = Xtr[:, :N]; th = np.zeros_like(V); th[:, g.nonref] = Xtr[:, N:]
    F = np.concatenate([V, th[:, g.nonref]], 1)
    sd = F.std(0) + 1e-9
    P = LedoitWolf().fit((F - F.mean(0)) / sd).precision_
    dirs = np.zeros((N, 2))
    for b in range(N):
        if b == g.ref:
            dirs[b] = [1.0, 0.0]; continue
        ii = [b, N + int(np.where(g.nonref == b)[0][0])]
        C = np.linalg.inv(P[np.ix_(ii, ii)])          # conditional covariance (standardised coords)
        w, U = np.linalg.eigh(C)
        e = U[:, -1] * sd[ii]                          # back to physical units (dV, dtheta)
        dirs[b] = e
    return dirs


def make_c(g, buses, vecs):
    """c from per-bus (dV, dtheta) vectors."""
    c = np.zeros(g.n)
    for b, (dv, dt) in zip(buses, vecs):
        c[b] += dv
        if b != g.ref:
            c[g.N + int(np.where(g.nonref == b)[0][0])] += dt
    return c


def tier_norm(vec, tier):
    mv, mt = TIER_MID[tier]
    return np.hypot(vec[0] / mv, vec[1] / mt)


def build_custom(g, X, idx, c_fn, rng, max_ch=60, rounds=8, gatt=None):
    """Generic AC-stealthy attack builder: z = h(x+c) + e, retained if WLS converges and J < tau (with grid g)."""
    K = len(idx); done = np.zeros(K, bool)
    out = dict(Z=np.zeros((K, g.m), np.float32), bus=np.zeros(K, int), xa=np.zeros((K, g.n)), J=np.zeros(K), base=idx.copy())
    for _ in range(rounds):
        todo = np.where(~done)[0]
        if len(todo) == 0:
            break
        C, B = [], []
        for k in todo:
            c, b = c_fn(k)
            C.append(c); B.append(b)
        C = np.array(C); Xb = X[idx[todo]]
        a = _h(g, Xb + C) - _h(g, Xb)
        a[np.abs(a) < 1e-10] = 0.0
        Za = _h(g, Xb) + rng.normal(size=a.shape) * g.sigma + a
        _, J, conv = g.wls(Za)
        vm = (Xb + C)[:, :g.N]
        ok = conv & (J < g.tau_bdd) & (vm.min(1) >= D.V_LIM_ATTACK[0]) & (vm.max(1) <= D.V_LIM_ATTACK[1]) & ((a != 0).sum(1) <= max_ch)
        acc = todo[ok]
        out["Z"][acc] = Za[ok]; out["bus"][acc] = np.array(B)[ok]; out["xa"][acc] = (Xb + C)[ok]; out["J"][acc] = J[ok]
        done[acc] = True
    keep = np.where(done)[0]
    return {k: v[keep] for k, v in out.items()}


def build(ds, g, rebuild=False):
    f = os.path.join(DATA_DIR, f"v2_{ds.system}_rep{ds.rep}.npz")
    if os.path.exists(f) and not rebuild:
        d = np.load(f)
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(20_000 + 131 * ds.rep + sum(map(ord, ds.system)))
    se = ChordSE(g, iters=3)
    dirs = anchor_directions(g, se, ds.normals("train"))
    elig = ds.d["eligible"]
    A = g.adjacency()
    out = {}
    load_buses = np.unique(g.bus_lookup[g.net.load.bus.values][g.net.load.p_mw.values > 1e-3])
    load_buses = np.array([b for b in load_buses if b != g.ref])
    p0 = np.zeros(g.N); np.add.at(p0, g.bus_lookup[g.net.load.bus.values], g.net.load.p_mw.values / g.baseMVA)
    for split, n_att in (("val", 500), ("test", 1500)):
        idx = ds.idx[split][:n_att] if split == "val" else ds.idx[split]
        X = ds.X
        # --- aligned single-bus
        for tier in ("M", "W"):
            v0, v1, a0, a1 = D.TIERS[tier]

            def c_al(k, tier=tier, v0=v0, v1=v1, a0=a0, a1=a1):
                b = int(rng.choice(elig))
                rnd = np.array([rng.uniform(v0, v1), np.deg2rad(rng.uniform(a0, a1))])
                rho = tier_norm(rnd, tier)
                e = dirs[b] / tier_norm(dirs[b], tier) * rho * rng.choice([-1, 1])
                return make_c(g, [b], [e]), b
            r = build_custom(g, X, idx, c_al, rng)
            for kk, vv in r.items():
                out[f"{split}_A_alig{tier}_{kk}"] = vv
        # --- coordinated multi-bus (random and aligned directions)
        for name, aligned in (("multiM", False), ("multialigM", True)):
            v0, v1, a0, a1 = D.TIERS["M"]

            def c_mu(k, aligned=aligned):
                b = int(rng.choice(elig))
                nb = [j for j in np.where(A[b])[0] if j != g.ref]
                kk = int(rng.integers(1, 3))
                grp = [b] + list(rng.choice(nb, min(kk, len(nb)), replace=False)) if nb else [b]
                rho = tier_norm(np.array([rng.uniform(v0, v1), np.deg2rad(rng.uniform(a0, a1))]), "M") / np.sqrt(len(grp))
                vecs = []
                for j in grp:
                    if aligned:
                        e = dirs[j] / tier_norm(dirs[j], "M")
                    else:
                        ang = rng.uniform(0, 2 * np.pi)
                        e = np.array([np.cos(ang) * TIER_MID["M"][0], np.sin(ang) * TIER_MID["M"][1]])
                        e = e / tier_norm(e, "M")
                    vecs.append(e * rho * rng.choice([-1, 1]))
                return make_c(g, grp, vecs), b
            r = build_custom(g, X, idx, c_mu, rng)
            for kk, vv in r.items():
                out[f"{split}_A_{name}_{kk}"] = vv
        # --- load redistribution (exact AC operating point mimicking a load shift)
        P, Q, V, pq = specs_from_state(g, X[idx])
        Pt, Qt = P.copy(), Q.copy()
        b1 = rng.choice(load_buses, len(idx)); b2 = np.array([rng.choice(load_buses[load_buses != x]) for x in b1])
        frac = rng.uniform(0.2, 0.5, len(idx))
        nri = {b: i for i, b in enumerate(g.nonref)}
        for k in range(len(idx)):
            dP = frac[k] * max(p0[b1[k]], p0[b2[k]])
            i1, i2 = nri[b1[k]], nri[b2[k]]
            pfr1 = Q[k, i1] / P[k, i1] if abs(P[k, i1]) > 1e-6 else 0.0
            pfr2 = Q[k, i2] / P[k, i2] if abs(P[k, i2]) > 1e-6 else 0.0
            Pt[k, i1] -= dP; Qt[k, i1] -= dP * pfr1   # more load at b1 (injection decreases)
            Pt[k, i2] += dP; Qt[k, i2] += dP * pfr2   # less load at b2
        xs, okpf = pf_solve(g, X[idx], Pt, Qt, V, pq)
        lr_c = xs - X[idx]
        sel = np.where(okpf)[0]
        r = build_custom(g, X, idx[sel], lambda k: (lr_c[sel][k], int(b1[sel][k])), rng, max_ch=g.m, rounds=1)
        for kk, vv in r.items():
            out[f"{split}_A_lr_{kk}"] = vv
    # --- operating-condition sets (test + validation recalibration pools)
    Xv, Xt = ds.X[ds.idx["val"]], ds.X[ds.idx["test"]]
    hv, ht = _h(g, Xv), _h(g, Xt)
    for k in NOISE:
        out[f"noise{k}_V"] = (hv + k * rng.normal(size=hv.shape) * g.sigma).astype(np.float32)
    att = {}
    for t in ("M", "W"):
        xa, base = ds.att(f"test_{t}", "xa"), ds.att(f"test_{t}", "base")
        att[t] = (_h(g, xa) - _h(g, ds.X[base]), _h(g, ds.X[base]))
    for q in MISS:
        sets = {"N": ht + rng.normal(size=ht.shape) * g.sigma, "V": hv + rng.normal(size=hv.shape) * g.sigma}
        for t, (a, hb) in att.items():
            sets[f"A_{t}"] = hb + a + rng.normal(size=a.shape) * g.sigma
        for kk, Z in sets.items():
            M = rng.random(Z.shape) < q
            out[f"missnan{q}_{kk}"] = np.where(M, np.nan, Z).astype(np.float32)
    # --- N-1 topology
    lines = pick_outage_lines(g, 3, seed=ds.rep)
    out["topo_lines"] = np.array(lines)
    for split, Xs, name in (("test", Xt, "N"), ("val", Xv, "V")):
        P, Q, V, pq = specs_from_state(g, Xs)
        lid = np.arange(len(Xs)) % 3
        Xo = np.zeros_like(Xs); okall = np.zeros(len(Xs), bool)
        for j, ln in enumerate(lines):
            go = outage_grid(g, ln)
            sel = np.where(lid == j)[0]
            xs, ok = pf_solve(go, Xs[sel], P[sel], Q[sel], V[sel], pq)
            vm = xs[:, :g.N]
            ok &= (vm.min(1) >= 0.85) & (vm.max(1) <= 1.15)
            Xo[sel] = xs; okall[sel] = ok
            Zo = _h(go, xs) + rng.normal(size=(len(sel), g.m)) * g.sigma
            out.setdefault(f"topo_{name}", np.zeros((len(Xs), g.m), np.float32))[sel] = Zo
            if name == "N":  # tier-M attacks under the outage, built with the outaged model
                v0, v1, a0, a1 = D.TIERS["M"]
                elig_o = elig

                def c_t(k):
                    b = int(rng.choice(elig_o))
                    vec = [rng.uniform(v0, v1) * rng.choice([-1, 1]), np.deg2rad(rng.uniform(a0, a1)) * rng.choice([-1, 1])]
                    return make_c(go, [b], [vec]), b
                Xtmp = np.zeros_like(ds.X); Xtmp[: len(sel)] = xs
                r = build_custom(go, Xtmp, np.arange(len(sel))[ok], c_t, rng)
                for kk, vv in r.items():
                    out.setdefault(f"topo_A_M_{kk}_parts", []).append(vv)
        out[f"topo_{name}_ok"] = okall
        out[f"topo_{name}_line"] = np.array(lines)[lid]
    parts = out.pop("topo_A_M_Z_parts"), out.pop("topo_A_M_bus_parts")
    out["topo_A_M"] = np.concatenate(parts[0]).astype(np.float32)
    out["topo_A_M_bus"] = np.concatenate(parts[1])
    for k in [k for k in list(out) if k.endswith("_parts")]:
        out.pop(k)
    out["topo_N"] = out["topo_N"][out["topo_N_ok"]]
    out["topo_V"] = out["topo_V"][out["topo_V_ok"]]
    out = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v) for k, v in out.items()}
    tmp = f[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, f)
    return out


if __name__ == "__main__":
    import argparse
    import time
    from common import Dataset
    from grid import Grid
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["ieee14p2", "ieee39p2", "ieee118p2", "ieee14", "ieee39", "ieee118"])
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    for s in a.systems:
        for r in a.reps:
            t0 = time.time()
            ds, g = Dataset(s, r), Grid(s)
            o = build(ds, g, a.rebuild)
            print(s, r, {k: (v.shape[0] if hasattr(v, "shape") and v.ndim else v) for k, v in o.items() if k.endswith("_Z") or k.startswith(("topo_N", "topo_A_M", "missnan0.05_N"))},
                  f"[{time.time() - t0:.0f}s]", flush=True)
