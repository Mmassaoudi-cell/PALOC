"""Real load / rooftop-PV profiles for protocol P2 from the Ausgrid Solar Home Electricity dataset
(300 NSW customers, half-hourly, 1 Jul 2012 - 30 Jun 2013). Consumption = GC + CL; generation = GG.
Output cache: data/ausgrid_profiles.npz with cons [C, T], gen [C, T], cap [C], time index (days, slots).
"""
import os

import numpy as np
import pandas as pd

from common import DATA_DIR

# Raw data folder (not distributed): set PALOC_RAW_DATA, default <repo>/raw_data
RAW_DIR = os.environ.get("PALOC_RAW_DATA", os.path.join(os.path.dirname(DATA_DIR), "raw_data"))
SRC = os.path.join(RAW_DIR, "2012-2013 Solar home electricity data v2.csv")
CACHE = os.path.join(DATA_DIR, "ausgrid_profiles.npz")


def load_profiles():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    df = pd.read_csv(SRC, skiprows=1)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    slots = [c for c in df.columns if ":" in c][:48]
    days = np.sort(df["date"].unique())
    cust = np.sort(df["Customer"].unique())
    C, D = len(cust), len(days)
    cons = np.zeros((C, D, 48)); gen = np.zeros((C, D, 48))
    ci = {c: i for i, c in enumerate(cust)}; di = {d: i for i, d in enumerate(days)}
    for cat, arr in (("GC", cons), ("CL", cons), ("GG", gen)):
        sub = df[df["Consumption Category"] == cat]
        idx_c = sub["Customer"].map(ci).values; idx_d = sub["date"].map(di).values
        np.add.at(arr, (idx_c, idx_d), sub[slots].fillna(0).values)
    cap = df.groupby("Customer")["Generator Capacity"].first().reindex(cust).values
    out = dict(cons=cons.reshape(C, -1).astype(np.float32), gen=gen.reshape(C, -1).astype(np.float32), cap=cap,
               day=np.repeat(np.arange(D), 48), slot=np.tile(np.arange(48), D),
               days=np.array([str(d)[:10] for d in days]))
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


if __name__ == "__main__":
    p = load_profiles()
    print({k: v.shape for k, v in p.items()}, p["days"][0], p["days"][-1])
