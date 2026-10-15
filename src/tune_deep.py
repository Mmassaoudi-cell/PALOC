"""Equal-budget validation tuning of the deep unsupervised baselines (same objective, systems and trial budget
as the proposed model): mean validation macro-F1 (val_mix, alpha=0.05) on ieee14p2 and ieee118p2, replicate 0.
python tune_deep.py --model SAE --trials 12
"""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import optuna
import torch

from common import ROOT, Dataset, evaluate_scores, set_seed
from grid import Grid
from registry import REGISTRY

SYSTEMS = ["ieee14p2", "ieee118p2"]


def space(t, model):
    p = dict(lr=t.suggest_float("lr", 1e-4, 3e-3, log=True), wd=t.suggest_float("wd", 1e-7, 1e-3, log=True),
             bs=t.suggest_categorical("bs", [64, 128, 256]), width=t.suggest_categorical("width", [0.5, 1.0, 2.0]))
    if model in ("SAE", "VAE"):
        p["latent"] = t.suggest_categorical("latent", [0.1, 0.25, 0.5, 1.0])
    if model == "GraphAE":
        p["gae_lat"] = t.suggest_categorical("gae_lat", [1, 2, 4])
    if model == "PIAE":
        p["w_phy"] = t.suggest_float("w_phy", 0.1, 10.0, log=True)
    if model == "DeepSVDD":
        p["rep"] = t.suggest_categorical("rep", [16, 32, 64])
    return p


def objective(t, model):
    p = space(t, model)
    f1s = []
    for i, s in enumerate(SYSTEMS):
        set_seed(0)
        ds, g = Dataset(s, 0), Grid(s)
        det = REGISTRY[f"{model}-t"](ds, g, seed=0, **p)
        sets = {k: v for k, v in ds.eval_sets().items() if k.startswith("val")}
        sc = {k: det.score(v) for k, v in sets.items()}
        f1 = [x for x in evaluate_scores(sc, alphas=(0.05,)) if x["set"] == "val_mix"][0]["macro_f1"]
        t.set_user_attr(f"{s}_f1", f1); f1s.append(f1)
        t.report(float(np.mean(f1s)), i)
        if t.should_prune():
            raise optuna.TrialPruned()
        torch.cuda.empty_cache()
    return float(np.mean(f1s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trials", type=int, default=12)
    a = ap.parse_args()
    d = os.path.join(ROOT, "logs", "tuning"); os.makedirs(d, exist_ok=True)
    st = optuna.create_study(study_name=f"deep_{a.model}", direction="maximize", storage=f"sqlite:///{d}/deep_{a.model}.db",
                             load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=5),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=4))
    if len(st.trials) == 0:  # source-paper settings as the first trial (cannot do worse than them)
        first = dict(lr=1e-3, wd=1e-5, bs=128, width=1.0)
        if a.model in ("SAE", "VAE"):
            first["latent"] = 1.0
        if a.model == "GraphAE":
            first["gae_lat"] = 2
        if a.model == "PIAE":
            first["w_phy"] = 1.0
        if a.model == "DeepSVDD":
            first["rep"] = 32
        st.enqueue_trial(first)
    st.optimize(lambda t: objective(t, a.model), n_trials=max(0, a.trials - len([t for t in st.trials if t.state.name in ("COMPLETE", "PRUNED")])))
    best = st.best_trial
    p = os.path.join(ROOT, "configs", "deep_hp.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur[a.model] = best.params
    json.dump(cur, open(p, "w"), indent=1)
    st.trials_dataframe().to_csv(os.path.join(d, f"deep_{a.model}_trials.csv"), index=False)
    print(a.model, best.value, best.params)
