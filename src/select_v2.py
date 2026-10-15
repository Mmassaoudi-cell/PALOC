"""Validation-only selection for PALOC v2 components (no test set is scored).
Objective: mean validation macro-F1 (alpha=0.05) over val_mix and the four validation detector-aware attack sets."""
import argparse
import os

import numpy as np
import pandas as pd

from common import RES_DIR, Dataset, metrics, set_seed, threshold
from grid import Grid
from registry import REGISTRY
import evalsets_v2

VAL_SETS = ["val_A", "val_A_aligM", "val_A_multiM", "val_A_multialigM", "val_A_lr"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["PALOC2", "PALOC2-nopair"])
    ap.add_argument("--systems", nargs="+", default=["ieee14p2", "ieee39p2", "ieee118p2"])
    ap.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    out = os.path.join(RES_DIR, "tables", "selection_v2.csv" if "ieee118p2" not in a.systems else "selection_v2_ieee118p2.csv")
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    for s in a.systems:
        for r in a.reps:
            ds, g = Dataset(s, r), Grid(s)
            v2 = evalsets_v2.build(ds, g)
            sets = {"val_N": ds.normals("val"), "val_A": ds.att("val_mix")}
            for k in ("aligM", "multiM", "multialigM", "lr"):
                sets[f"val_A_{k}"] = v2[f"val_A_{k}_Z"]
            for m in a.methods:
                if any(x["method"] == m and x["system"] == s and x["rep"] == r for x in rows):
                    continue
                set_seed(r)
                det = REGISTRY[m](ds, g, seed=r)
                sc = {k: det.score(v) for k, v in sets.items()}
                tau = threshold(sc["val_N"], 0.05)
                for k in VAL_SETS:
                    mt = metrics(sc["val_N"], sc[k], tau)
                    rows.append(dict(method=m, system=s, rep=r, set=k, macro_f1=mt["macro_f1"], auc=mt["auc"], tpr=mt["tpr"]))
                pd.DataFrame(rows).to_csv(out, index=False)
                print(m, s, r, {k: round(x["macro_f1"], 4) for x in rows[-len(VAL_SETS):] for k in [x["set"]]}, flush=True)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["method"], columns="set", values="macro_f1").round(4))
    print(df.groupby("method").macro_f1.mean().round(4))
