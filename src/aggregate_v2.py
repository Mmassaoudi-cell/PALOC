"""Statistics for the v2 evaluation.
Primary evidence: per-system paired differences (proposed - benchmark) over replicates with 95% t-intervals, paired
Cohen's d_z, and paired t-test p-values Holm-corrected across benchmarks within each system (exact Wilcoxon signed-rank
p-values, also Holm-corrected, are reported as a non-parametric check; with n=5 their minimum attainable value is 0.0625).
Secondary: Wilcoxon pooled over all (system, replicate) pairs of a protocol (Holm-corrected).
Families: 'nominal' (tier-mean over S/M/W), 'weak' (tier W), 'adversarial' (mean over aligM, aligW, multiM, multialigM, lr).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import RES_DIR, ROOT

PROPOSED = "PALOC2"
UNSUP = ["BDD", "PCA", "IForest", "OCSVM", "kNN", "StateMaha", "BayesGLRT", "SAE-t", "VAE-t", "GraphAE-t", "PIAE-t", "DeepSVDD-t", "PIBAE-t"]
SUP = ["Sup-LR", "Sup-RF", "Sup-ET", "Sup-HGB", "Sup-XGB", "Sup-LGBM", "Sup-CatBoost", "Sup-MLP", "Sup-FTT", "Sup-GNN", "SupS-XGB", "SupS-LGBM", "SupS-MLP"]
FAM = {"nominal": ["S", "M", "W"], "weak": ["W"], "adversarial": ["aligM", "aligW", "multiM", "multialigM", "lr"]}
TABLES = os.path.join(RES_DIR, "tables")


def load(tags=("v2", "main")):
    fs = []
    for t in tags:
        fs += glob.glob(os.path.join(RES_DIR, "raw", t, "*.csv"))
    dfs = []
    for f in fs:
        d = pd.read_csv(f)
        d["tag"] = os.path.basename(os.path.dirname(f))
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    if "calib" not in df:
        df["calib"] = "nominal"
    df["calib"] = df["calib"].fillna("nominal")
    df["protocol"] = np.where(df.system.str.endswith("p2"), "P2", "P1")
    df["base"] = df.system.str.replace("p2", "", regex=False)
    # 'main' tag only contributes methods absent from 'v2' (the raw-feature supervised references)
    v2m = set(df[df.tag == "v2"].method)
    df = df[(df.tag == "v2") | (~df.method.isin(v2m))]
    df = df[df.system.str.fullmatch(r"ieee\d+(p2)?")]
    return df


def family_scores(df, fam, metric="macro_f1", alpha=0.05):
    d = df[(df.alpha == alpha) & (df.calib == "nominal") & df.set.isin(FAM[fam])]
    g = d.groupby(["protocol", "base", "method", "rep"])
    out = g[metric].mean().reset_index()
    n = g.set.nunique().reset_index(name="nsets")
    out = out.merge(n, on=["protocol", "base", "method", "rep"])
    return out[out.nsets == len(FAM[fam])].drop(columns="nsets")


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def per_system(T, benches, metric="macro_f1"):
    rows = []
    for (proto, base), g in T.groupby(["protocol", "base"]):
        P = g[g.method == PROPOSED].set_index("rep")[metric]
        recs = []
        for b in benches:
            B = g[g.method == b].set_index("rep")[metric]
            j = pd.concat([P, B], axis=1, keys=["p", "b"]).dropna()
            if len(j) < 3:
                continue
            d = (j.p - j.b).values
            n = len(d); sd = d.std(ddof=1)
            h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if sd > 0 else 0.0
            pw = stats.wilcoxon(j.p, j.b).pvalue if np.any(d != 0) else 1.0
            pt = stats.ttest_rel(j.p, j.b).pvalue if sd > 0 else (0.0 if np.all(d != 0) else 1.0)
            recs.append(dict(protocol=proto, system=base, benchmark=b, n=n, proposed=j.p.mean(), bench=j.b.mean(), diff=d.mean(),
                             ci_lo=d.mean() - h, ci_hi=d.mean() + h, dz=d.mean() / sd if sd > 0 else np.inf * np.sign(d.mean()),
                             p_ttest=pt, p_wilcoxon=pw))
        if not recs:
            continue
        adj = holm([r["p_ttest"] for r in recs])
        adjw = holm([r["p_wilcoxon"] for r in recs])
        for r, q, qw in zip(recs, adj, adjw):
            r["p_holm"] = q
            r["p_wilcoxon_holm"] = qw
            r["outcome"] = ("win" if r["diff"] > 0 else "loss") if (q < 0.05 and r["ci_lo"] * r["ci_hi"] > 0) else ("win*" if r["ci_lo"] > 0 else ("loss*" if r["ci_hi"] < 0 else "tie"))
            rows.append(r)
    return pd.DataFrame(rows)


def pooled(T, benches, metric="macro_f1"):
    rows = []
    for proto, g in T.groupby("protocol"):
        P = g[g.method == PROPOSED].set_index(["base", "rep"])[metric]
        recs = []
        for b in benches:
            B = g[g.method == b].set_index(["base", "rep"])[metric]
            j = pd.concat([P, B], axis=1, keys=["p", "b"]).dropna()
            if len(j) < 5:
                continue
            d = (j.p - j.b).values
            recs.append(dict(protocol=proto, benchmark=b, n_pairs=len(d), diff=d.mean(), p=stats.wilcoxon(j.p, j.b).pvalue if np.any(d != 0) else 1.0,
                             min_attainable_p=2 * 0.5 ** len(d)))
        adj = holm([r["p"] for r in recs])
        for r, q in zip(recs, adj):
            r["p_holm"] = q
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    os.makedirs(TABLES, exist_ok=True)
    df = load()
    df.to_csv(os.path.join(TABLES, "v2_all_seed_level.csv"), index=False)
    allW = []
    for fam in FAM:
        T = family_scores(df, fam)
        T.to_csv(os.path.join(TABLES, f"v2_{fam}_seed_level.csv"), index=False)
        for grp, benches in (("unsupervised", UNSUP), ("supervised", SUP)):
            ps = per_system(T, benches); ps.insert(0, "family", fam); ps.insert(1, "group", grp)
            allW.append(ps)
            po = pooled(T, benches); po.insert(0, "family", fam); po.insert(1, "group", grp)
            po.to_csv(os.path.join(TABLES, f"v2_pooled_{fam}_{grp}.csv"), index=False)
    W = pd.concat(allW, ignore_index=True)
    W.to_csv(os.path.join(ROOT, "BENCHMARK_WTL.csv"), index=False)
    for (fam, grp, proto), g in W.groupby(["family", "group", "protocol"]):
        c = g.outcome.value_counts().to_dict()
        print(f"{fam:12s} {grp:13s} {proto}: wins={c.get('win', 0)} (+{c.get('win*', 0)} CI-only) ties={c.get('tie', 0)} losses={c.get('loss', 0)} (+{c.get('loss*', 0)} CI-only) of {len(g)}")
    return W


if __name__ == "__main__":
    if len(sys.argv) > 1:
        PROPOSED = sys.argv[1]
    main()
