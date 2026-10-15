"""Protocol P2 ("realistic operating envelope") data generation.

Differences from the paper-faithful protocol P1 (datagen.py), all fixed a priori:
  * per-bus load shapes = normalised aggregates of 15 real Ausgrid customers (half-hourly, 2012-13)
    f_j(t) = 1 + kappa (shape_j(t) - 1), kappa = 0.35
  * rooftop PV at ~20% of load buses: capacity 0.4 P0_j x real normalised PV output of 15 customers
  * per-snapshot load power factor ~ U[0.90, 0.99]
  * generator voltage set-points jittered U[-0.01, +0.01] p.u.
  * distributed slack (participation proportional to generator Pmax) and ENFORCED generator Q-limits
    (PV->PQ switching; a regime nonlinearity the source paper explicitly leaves unmodelled)
  * TEMPORAL split by whole weeks (70/15/15 % of weeks per replicate), no timestamp shared across splits
Attacks: identical generator to P1 (datagen.build_attacks).
Usage: python datagen_p2.py --system ieee14 --reps 0 1 2 3 4
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandapower as pp

import datagen as D
from grid import Grid
from profiles import RAW_DIR, load_profiles

KAPPA, GROUP, PV_FRAC, PV_CAP = 0.35, 15, 0.2, 0.4
# case39's nominal loading sits at its generator Q-limits; with enforced limits the realistic envelope
# is AC-infeasible in ~50% of hours, so its base load is scaled to 85% (decided before any P2 model run).
BASE_LOAD = {"ieee14": 1.0, "ieee39": 0.85, "ieee118": 1.0}
_G = {}


def _init(system):
    g = Grid(system)
    net = g.net
    net.gen["slack_weight"] = net.gen.max_p_mw.clip(lower=1.0)
    net.ext_grid["slack_weight"] = float(net.gen.max_p_mw.clip(lower=1.0).mean())
    b = BASE_LOAD[system]
    _G.update(g=g, p0=b * net.load.p_mw.values.copy(), q0=b * net.load.q_mvar.values.copy(), vm0=net.gen.vm_pu.values.copy())


def _solve(tasks):
    g = _G["g"]; net = g.net
    out = []
    for (f, pf, vmj, pv_bus, pv_mw) in tasks:
        p = _G["p0"] * f
        q = np.where(_G["p0"] > 0, p * np.tan(np.arccos(pf)), _G["q0"] * f)
        net.load["p_mw"] = p; net.load["q_mvar"] = q
        net.gen["vm_pu"] = _G["vm0"] + vmj
        net.sgen.drop(net.sgen.index, inplace=True)
        for b, mw in zip(pv_bus, pv_mw):
            pp.create_sgen(net, int(b), p_mw=float(mw), q_mvar=0.0)
        ok = False
        for init in ("results", "dc", "flat"):  # never warm-start from a diverged solution
            try:
                pp.runpp(net, init=init, max_iteration=50, distributed_slack=True, enforce_q_lims=True)
                ok = net.converged
            except Exception:
                ok = False
            if ok:
                break
        if not ok:
            out.append(None); continue
        x = g.state_from_net(net)
        vm = x[: g.N]
        out.append(x if (vm.min() >= D.V_LIM_NORMAL[0] and vm.max() <= D.V_LIM_NORMAL[1]) else None)
    return out


def load_isone():
    import pandas as pd
    df = pd.read_csv(os.path.join(RAW_DIR, "isone_load.csv"), sep=";")
    d = pd.to_datetime(df["date"], format="%d/%m/%Y")
    day = ((d - d.min()).dt.days).values
    return dict(load=df["load"].values.astype(float), day=day, doy=d.dt.dayofyear.values - 1, hour=df["hour"].values - 1)


def generate(system, rep, procs=6, kappa=KAPPA, pv_frac=PV_FRAC, source="ausgrid", suffix=""):
    t0 = time.time()
    rng = np.random.default_rng({"ieee14": 5000, "ieee39": 6000, "ieee118": 7000}[system] + rep)
    g = Grid(system)
    prof = load_profiles()
    C, T = prof["cons"].shape
    net = g.net
    nl, ngen = len(net.load), len(net.gen)
    # per-bus customer groups -> normalised shapes
    shapes = np.zeros((nl, T));
    for j in range(nl):
        grp = rng.choice(C, GROUP, replace=False)
        s = prof["cons"][grp].sum(0); shapes[j] = s / s.mean()
    tmap = np.arange(T)                     # profile time index -> PV time index
    week_of = prof["day"] // 7
    if source == "isone":                   # transmission-level shapes: ISO-NE hourly system load (2003-2014)
        iso = load_isone()
        Ti = len(iso["load"])
        base = iso["load"] / iso["load"].mean()
        shapes = np.zeros((nl, Ti))
        for j in range(nl):
            sh = int(rng.integers(-1, 2))                     # +-1 h regional offset
            eps = np.zeros(Ti); phi, sig = 0.9, 0.03 * np.sqrt(1 - 0.81)
            noise = rng.normal(size=Ti) * sig
            for t in range(1, Ti):
                eps[t] = phi * eps[t - 1] + noise[t]
            shapes[j] = np.roll(base, sh) * (1 + eps)
        T = Ti
        tmap = (iso["doy"] * 48 + iso["hour"] * 2) % prof["gen"].shape[1]   # same time of year / day for PV
        week_of = iso["day"] // 7
    F = np.clip(1 + kappa * (shapes - 1), 0.3, 2.0)
    load_bus = net.load.bus.values
    pv_idx = rng.choice(nl, int(round(pv_frac * nl)), replace=False)
    pvshape = np.zeros((len(pv_idx), T))
    pvcust = np.where(prof["cap"] > 0)[0]
    for k in range(len(pv_idx)):
        grp = rng.choice(pvcust, GROUP, replace=False)
        pvshape[k] = (prof["gen"][grp].sum(0) / (0.5 * prof["cap"][grp].sum()))[tmap]  # kWh per half-hour / (0.5 h x kWp)
    pvshape = np.clip(pvshape, 0, 1)
    p0 = BASE_LOAD[system] * net.load.p_mw.values
    pv_cap = PV_CAP * np.maximum(p0[pv_idx], 1.0)
    pv_bus = load_bus[pv_idx]
    # temporal split by weeks
    week = week_of
    W = week.max() + 1
    wp = rng.permutation(W)
    ntr, nva = int(round(0.70 * W)), int(round(0.15 * W))
    wsplit = np.empty(W, dtype="<U5"); wsplit[wp[:ntr]] = "train"; wsplit[wp[ntr:ntr + nva]] = "val"; wsplit[wp[ntr + nva:]] = "test"
    need = {"train": D.N_TRAIN, "val": D.N_VAL, "test": D.N_NORMAL - D.N_TRAIN - D.N_VAL}
    X, split, tstamp = [], [], []
    with Pool(procs, initializer=_init, initargs=(system,)) as pool:
        for s, n in need.items():
            cand = rng.permutation(np.where(wsplit[week] == s)[0])
            got, ptr = [], 0
            while len(got) < n:
                if ptr >= len(cand):
                    raise RuntimeError(f"{system} {s}: only {len(got)}/{n} feasible snapshots")
                tt = cand[ptr:ptr + int((n - len(got)) * 1.15) + 32]; ptr += len(tt)
                tasks = [(F[:, t], rng.uniform(0.90, 0.99, nl), rng.uniform(-0.01, 0.01, ngen), pv_bus, pv_cap * pvshape[:, t]) for t in tt]
                res = []
                for r in pool.map(_solve, np.array_split(np.array(tasks, dtype=object), procs * 4)):
                    res += r
                got += [(x, t) for x, t in zip(res, tt) if x is not None]
            X += [x for x, _ in got[:n]]; tstamp += [t for _, t in got[:n]]; split += [s] * n
    X, split, tstamp = np.array(X), np.array(split), np.array(tstamp)
    E = rng.normal(size=(len(X), g.m)) * g.sigma
    Z = D.h_np(g, X) + E
    xh, J, conv = g.wls(Z)
    elig = D.eligible_buses(g, g.state_from_net(g.net))
    out = dict(X=X, Z=Z.astype(np.float32), split=split, J=J, wls_conv=conv, eligible=elig, tries=0, tstamp=tstamp)
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    E2 = rng.normal(size=(len(X), g.m)) * g.sigma
    att = {}
    for t in D.TIERS:
        att[f"test_{t}"] = D.build_attacks(g, X, E2, idx["test"], [t] * len(idx["test"]), elig, rng)
    rr = lambda n: np.array(list(D.TIERS))[np.arange(n) % 3]
    att["val_mix"] = D.build_attacks(g, X, E2, idx["val"], rr(len(idx["val"])), elig, rng)
    att["train_mix"] = D.build_attacks(g, X, E2, idx["train"], rr(len(idx["train"])), elig, rng)
    att["test_generic"] = D.build_attacks(g, X, E2, idx["test"], ["S"] * len(idx["test"]), elig, rng, kind="generic")
    att["test_gray"] = D.build_attacks(g, X, E2, idx["test"], ["S"] * len(idx["test"]), elig, rng, kind="gray", g_att=D.perturbed_grid(g, rng))
    for k, v in att.items():
        for kk, vv in v.items():
            out[f"A_{k}_{kk}"] = vv
    np.savez_compressed(os.path.join(D.DATA_DIR, f"{system}p2{suffix}_rep{rep}.npz"), **out)
    print(f"{system}p2{suffix} rep{rep}: J_mean={J.mean():.1f} BDD_FPR={(J > g.tau_bdd).mean():.3f} Vrange=[{X[:, :g.N].min():.3f},{X[:, :g.N].max():.3f}] "
          + " ".join(f"{k}:{len(v['bus'])}" for k, v in att.items()) + f" [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--reps", type=int, nargs="+", default=[0])
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--pv", type=float, default=PV_FRAC)
    ap.add_argument("--source", default="ausgrid")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    for r in a.reps:
        generate(a.system, r, kappa=a.kappa, pv_frac=a.pv, source=a.source, suffix=a.suffix)
