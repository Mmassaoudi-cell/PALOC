"""Aggregate seed-level results -> summary CSVs, statistical tests, win/tie/loss table.

Primary metric: macro-F1 at the conformal 95% threshold (alpha = 0.05), averaged over the three
stealthy tiers {S, M, W} of a system ("tier-mean macro-F1"); also reported per tier.
Statistics (proposed vs. each benchmark):
  * per system: paired t-test over the 5 replicates
  * pooled: Wilcoxon signed-rank over all (system, replicate) pairs of a protocol (n = 15)
  * Holm correction across benchmarks within each (protocol, test family)
  * effect sizes: paired Cohen's d_z and matched-pairs rank-biserial correlation
Classification: superior (Holm p<0.05, diff>0), inferior (Holm p<0.05, diff<0), else
"practically superior" if diff >= 0.01, else "indistinguishable".
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import RES_DIR

TABLES = os.path.join(RES_DIR, "tables")
UNSUP = ["BDD", "PCA", "IForest", "OCSVM", "kNN", "StateMaha", "SAE", "VAE", "GraphAE", "PIAE", "PIBAE", "DeepSVDD"]
SUP = ["Sup-LR", "Sup-RF", "Sup-ET", "Sup-HGB", "Sup-XGB", "Sup-LGBM", "Sup-CatBoost", "Sup-MLP", "Sup-FTT", "Sup-GNN"]
PROTO = lambda s: "P2" if s.endswith("p2") else "P1"


def load(tags=("main", "final")):
    fs = []
    for t in tags:
        fs += glob.glob(os.path.join(RES_DIR, "raw", t, "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["protocol"] = df.system.map(PROTO)
    df["base"] = df.system.str.replace("p2", "", regex=False)
    return df


def tiermean(df, metric="macro_f1", alpha=0.05):
    d = df[(df.alpha == alpha) & df.set.isin(["S", "M", "W"])]
    return d.groupby(["protocol", "base", "system", "method", "rep"])[metric].mean().rename(metric).reset_index()


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def summary(df):
    rows = []
    for (proto, sysn, m, st, a), g in df.groupby(["protocol", "system", "method", "set", "alpha"]):
        for metric in ["auc", "prauc", "macro_f1", "weighted_f1", "bal_acc", "tpr", "fpr", "fnr", "recall_attack", "recall_normal"]:
            x = g[metric].values
            lo, hi = ci95(x)
            rows.append(dict(protocol=proto, system=sysn, method=m, set=st, alpha=a, metric=metric, n=len(x),
                             mean=x.mean(), std=x.std(ddof=1) if len(x) > 1 else np.nan, median=np.median(x), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def wtl(tm, proposed, benches, metric="macro_f1"):
    rows = []
    for proto, g in tm.groupby("protocol"):
        P = g[g.method == proposed].set_index(["system", "rep"])[metric]
        recs = []
        for b in benches:
            B = g[g.method == b].set_index(["system", "rep"])[metric]
            j = P.to_frame("p").join(B.to_frame("b"), how="inner")
            if j.empty:
                continue
            d = (j.p - j.b).values
            try:
                pw = stats.wilcoxon(j.p, j.b, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
            except ValueError:
                pw = 1.0
            per_sys = {}
            for sysn, js in j.groupby(level=0):
                ds = (js.p - js.b).values
                pt = stats.ttest_rel(js.p, js.b).pvalue if (len(ds) > 1 and ds.std() > 0) else (0.0 if np.all(ds != 0) and len(ds) > 1 else 1.0)
                per_sys[sysn] = (ds.mean(), pt)
            recs.append(dict(protocol=proto, benchmark=b, n_pairs=len(d), mean_diff=d.mean(), median_diff=np.median(d),
                             proposed_mean=j.p.mean(), benchmark_mean=j.b.mean(),
                             cohen_dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.inf * np.sign(d.mean()),
                             rank_biserial=rank_biserial(d), p_wilcoxon=pw, per_sys=per_sys))
        if not recs:
            continue
        adj = holm([r["p_wilcoxon"] for r in recs])
        # per-system Holm over benchmarks
        systems = sorted({s for r in recs for s in r["per_sys"]})
        for s in systems:
            ps = [r["per_sys"].get(s, (np.nan, 1.0))[1] for r in recs]
            a2 = holm(ps)
            for r, q in zip(recs, a2):
                if s in r["per_sys"]:
                    r.setdefault("per_sys_adj", {})[s] = (r["per_sys"][s][0], q)
        for r, q in zip(recs, adj):
            r["p_wilcoxon_holm"] = q
            w = t = l = 0
            for s, (dm, qs) in r.get("per_sys_adj", {}).items():
                if qs < 0.05 and dm > 0:
                    w += 1
                elif qs < 0.05 and dm < 0:
                    l += 1
                else:
                    t += 1
            r["systems_won"], r["systems_tied"], r["systems_lost"] = w, t, l
            if q < 0.05 and r["mean_diff"] > 0:
                r["overall"] = "statistically superior"
            elif q < 0.05 and r["mean_diff"] < 0:
                r["overall"] = "inferior"
            elif r["mean_diff"] >= 0.01:
                r["overall"] = "practically superior"
            elif r["mean_diff"] <= -0.01:
                r["overall"] = "practically inferior"
            else:
                r["overall"] = "statistically indistinguishable"
            r["per_system"] = "; ".join(f"{s}:{dm:+.4f}(p_holm={qs:.3g})" for s, (dm, qs) in r.get("per_sys_adj", {}).items())
            del r["per_sys"]; r.pop("per_sys_adj", None)
            rows.append(r)
    return pd.DataFrame(rows)


def main(proposed):
    os.makedirs(TABLES, exist_ok=True)
    df = load()
    df.to_csv(os.path.join(TABLES, "all_seed_level.csv"), index=False)
    summary(df).to_csv(os.path.join(TABLES, "summary_all.csv"), index=False)
    tm = tiermean(df)
    tm.to_csv(os.path.join(TABLES, "tiermean_macro_f1_seed_level.csv"), index=False)
    tma = tiermean(df, "auc")
    out = []
    for fam, benches in (("unsupervised", UNSUP), ("supervised_privileged", SUP)):
        for metric, T in (("macro_f1", tm), ("auc", tma)):
            w = wtl(T, proposed, benches, metric)
            w.insert(0, "family", fam); w.insert(1, "metric", "tier-mean " + metric)
            out.append(w)
        # tier-W only (difficult regime)
        dW = df[(df.alpha == 0.05) & (df.set == "W")].groupby(["protocol", "system", "method", "rep"]).macro_f1.mean().reset_index()
        w = wtl(dW, proposed, benches, "macro_f1")
        w.insert(0, "family", fam); w.insert(1, "metric", "tier-W macro_f1")
        out.append(w)
    W = pd.concat(out, ignore_index=True)
    W.to_csv(os.path.join(RES_DIR, "..", "BENCHMARK_WTL.csv"), index=False)
    for (fam, met, proto), g in W.groupby(["family", "metric", "protocol"]):
        c = g.overall.value_counts().to_dict()
        print(f"{fam:22s} {met:22s} {proto}: superior={c.get('statistically superior', 0)} practical={c.get('practically superior', 0)} "
              f"ties={c.get('statistically indistinguishable', 0)} prac_inferior={c.get('practically inferior', 0)} inferior={c.get('inferior', 0)}")
    return W


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "PROPOSED")
