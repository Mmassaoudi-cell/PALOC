"""Stage-3 validation-only tuning of the top candidates (Optuna TPE + median pruning).
Objective: mean VALIDATION macro-F1 (val_mix, conformal 95% threshold) over tuning systems, replicate 0.
TEST data are never scored. Study storage: logs/tuning/<candidate>.db
python tune_candidates.py --cand BSP --trials 16
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


def space(trial, cand):
    p = dict(lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
             wd=trial.suggest_float("wd", 1e-6, 1e-3, log=True),
             agg=trial.suggest_categorical("agg", ["max", "maxcal", "top3"]),
             fuse=trial.suggest_categorical("fuse", [False, True]),
             epochs=100, patience=15)
    if cand == "BSP":
        p.update(r=trial.suggest_categorical("r", [32, 64, 128]),
                 hidden=trial.suggest_categorical("hidden", [64, 128, 256]),
                 depth=trial.suggest_int("depth", 1, 3),
                 emb=trial.suggest_categorical("emb", [0, 8, 16]))
    elif cand in ("PGMP_AM", "PGMP_MLP"):
        p.update(hidden=trial.suggest_categorical("hidden", [128, 256, 512]),
                 dropout=trial.suggest_categorical("dropout", [0.0, 0.1]))
    elif cand in ("PGMP_A", "PGMP"):
        p.update(d=trial.suggest_categorical("d", [32, 48, 64]), layers=trial.suggest_int("layers", 2, 4),
                 bus_emb=trial.suggest_categorical("bus_emb", [0, 8]))
    return p


def evaluate(cand, system, p, seed=0):
    set_seed(seed)
    ds, g = Dataset(system, 0), Grid(system)
    det = REGISTRY[cand](ds, g, seed=seed, **p)
    sets = {k: v for k, v in ds.eval_sets().items() if k.startswith("val")}
    sc = {k: det.score(v) for k, v in sets.items()}
    r = {x["set"]: x for x in evaluate_scores(sc, alphas=(0.05,))}
    return r["val_mix"]["macro_f1"], r["val_mix"]["auc"], r["val_W"]["macro_f1"]


def objective(trial, cand):
    p = space(trial, cand)
    f1s = []
    for i, s in enumerate(SYSTEMS):
        f1, auc, f1w = evaluate(cand, s, p)
        trial.set_user_attr(f"{s}_f1", f1); trial.set_user_attr(f"{s}_auc", auc); trial.set_user_attr(f"{s}_f1W", f1w)
        f1s.append(f1)
        trial.report(float(np.mean(f1s)), i)
        if trial.should_prune():
            raise optuna.TrialPruned()
        torch.cuda.empty_cache()
    return float(np.mean(f1s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--trials", type=int, default=16)
    a = ap.parse_args()
    d = os.path.join(ROOT, "logs", "tuning"); os.makedirs(d, exist_ok=True)
    storage = f"sqlite:///{d}/{a.cand}.db"
    try:  # reseed on resume so a restarted study does not replay its earlier random proposals
        n_prev = len(optuna.load_study(study_name=a.cand, storage=storage).trials)
    except KeyError:
        n_prev = 0
    study = optuna.create_study(study_name=a.cand, direction="maximize", storage=storage,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=n_prev, n_startup_trials=6),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4))
    # enqueue the screening default so tuning can only improve on it
    if len(study.trials) == 0:
        default = dict(lr=2e-3, wd=1e-4 if a.cand == "BSP" else 1e-5, agg="max", fuse=False)
        if a.cand == "BSP":
            default.update(r=64, hidden=128, depth=2, emb=8)
        elif a.cand in ("PGMP_AM", "PGMP_MLP"):
            default.update(hidden=256, dropout=0.0)
        else:
            default.update(d=48, layers=3, bus_emb=8)
        study.enqueue_trial(default)
    study.optimize(lambda t: objective(t, a.cand), n_trials=a.trials - len([t for t in study.trials if t.state.is_finished()]))
    df = study.trials_dataframe()
    df.to_csv(os.path.join(d, f"{a.cand}_trials.csv"), index=False)
    best = study.best_trial
    json.dump(dict(value=best.value, params=best.params, attrs=best.user_attrs), open(os.path.join(d, f"{a.cand}_best.json"), "w"), indent=1)
    print(a.cand, "best", best.value, best.params, best.user_attrs)
