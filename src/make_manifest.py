"""DATA_SPLIT_MANIFEST.csv: per (system, replicate, split) sample counts, attack counts, temporal coverage (P2),
exact cross-split duplicates and nearest-neighbour (near-duplicate) distances between TEST and TRAIN normals."""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import DATA_DIR, ROOT

rows = []
for f in sorted(glob.glob(os.path.join(DATA_DIR, "ieee*_rep*.npz"))):
    name = os.path.basename(f)[:-4]
    system, rep = name.split("_rep")
    d = np.load(f)
    split = d["split"]
    Z = d["Z"].astype(np.float64)
    tr = split == "train"
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
    Zn = (Z - mu) / sd
    nn = NearestNeighbors(n_neighbors=1).fit(Zn[tr])
    # typical within-train NN distance for reference
    d_tt = NearestNeighbors(n_neighbors=2).fit(Zn[tr]).kneighbors(Zn[tr][:1000])[0][:, 1]
    for s in ("train", "val", "test"):
        m = split == s
        r = dict(system=system, protocol="P2" if system.endswith("p2") else "P1", replicate=int(rep), split=s,
                 n_normal=int(m.sum()))
        if s != "train":
            dist = nn.kneighbors(Zn[m])[0][:, 0]
            r.update(exact_duplicates_with_train=int((dist < 1e-9).sum()), nn_dist_to_train_min=float(dist.min()),
                     nn_dist_to_train_median=float(np.median(dist)), train_internal_nn_median=float(np.median(d_tt)))
        if "tstamp" in d.files:
            ts = d["tstamp"][m]
            r.update(n_unique_timestamps=int(len(np.unique(ts))), first_halfhour=int(ts.min()), last_halfhour=int(ts.max()),
                     n_weeks=int(len(np.unique(ts // (48 * 7)))))
        for k in d.files:
            if k.startswith(f"A_{s}_") and k.endswith("_bus"):
                r[f"attacks_{k[2:-4]}"] = int(len(d[k]))
        rows.append(r)
    if "tstamp" in d.files:  # temporal leakage check: no week shared between splits
        wk = {s: set(np.unique(d["tstamp"][split == s] // (48 * 7))) for s in ("train", "val", "test")}
        assert not (wk["train"] & wk["test"]) and not (wk["train"] & wk["val"]) and not (wk["val"] & wk["test"]), name
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "DATA_SPLIT_MANIFEST.csv"), index=False)
print(pd.DataFrame(rows).groupby(["protocol", "split"])[["n_normal", "exact_duplicates_with_train"]].sum())
