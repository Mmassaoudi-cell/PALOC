"""Accuracy of the fixed-gain chord estimator vs. full Gauss-Newton WLS (validation normals, replicate 0)."""
import os, time
import numpy as np, pandas as pd
from common import Dataset, RES_DIR
from grid import Grid
from physics import ChordSE
rows = []
for s in ["ieee14", "ieee39", "ieee118", "ieee14p2", "ieee39p2", "ieee118p2"]:
    ds, g = Dataset(s, 0), Grid(s)
    Z, X = ds.normals("val"), ds.X[ds.idx["val"]]
    t = time.time(); xw, Jw, cw = g.wls(Z); tw = time.time() - t
    for K in (1, 2, 3):
        se = ChordSE(g, iters=K); t = time.time(); xc, Jc = se.estimate(Z); tc = time.time() - t
        rows.append(dict(system=s, K=K, mae_chord_vs_true=np.abs(xc - X).mean(), mae_wls_vs_true=np.abs(xw - X).mean(),
                         max_abs_chord_minus_wls=np.abs(xc - xw).max(), mean_J_ratio=(Jc / Jw).mean(), wls_converged=cw.mean(),
                         gpu_us_per_sample_chord=1e6 * tc / len(Z), gpu_us_per_sample_wls=1e6 * tw / len(Z)))
        print(rows[-1], flush=True)
os.makedirs(os.path.join(RES_DIR, "tables"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, "tables", "se_accuracy.csv"), index=False)
