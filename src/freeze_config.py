"""Freeze FINAL_MODEL_CONFIG.yaml from the validation-only Stage-3 tuning result of the selected candidate.
python freeze_config.py --cand BSP --name PALOC
"""
import argparse
import datetime
import hashlib
import json
import os

import yaml

from common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--cand", required=True)
ap.add_argument("--name", default="PALOC")
a = ap.parse_args()
best = json.load(open(os.path.join(ROOT, "logs", "tuning", f"{a.cand}_best.json")))
p = dict(best["params"])
p.update(epochs=150, patience=20)  # full training budget for final runs (tuning used 100/15)
# chord iterations: smallest K whose validation-normal state MAE is within 6% of full WLS on every system
# (results/tables/se_accuracy.csv; K=2 under-converges on IEEE-39). Validation normals only, no attack labels.
p["se_iters"] = 3
code_hash = hashlib.sha256(open(os.path.join(ROOT, "src", "candidates.py"), "rb").read()
                           + open(os.path.join(ROOT, "src", "physics.py"), "rb").read()
                           + open(os.path.join(ROOT, "src", "grid.py"), "rb").read()).hexdigest()
cfg = {
    "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "model_name": a.name,
    "registry_candidate": a.cand,
    "selection": {"stage": "Stage-3 Optuna TPE, validation only", "objective": "mean validation macro-F1 (val_mix, alpha=0.05) on ieee14p2 & ieee118p2 rep0",
                  "best_validation_objective": best["value"], "per_system_validation": best["attrs"]},
    "architecture": {
        "state_estimator": {"type": "fixed-gain chord Gauss-Newton AC SE", "iterations": p.get("se_iters", 2),
                            "gain": "G0=(H0^T W H0)^-1 H0^T W at pandapower base case", "W": "R^-1 (sigma_V=0.002, sigma_PQ=0.01 p.u.)"},
        "anchor": {"type": "Ledoit-Wolf precision on standardised per-bus states (TRAIN normals)", "statistic": "r_b = Lambda_bb^-1 [Lambda (s-mu)]_b", "whitening": True},
        "blind_spot_network": {k: p[k] for k in ("r", "hidden", "depth", "emb") if k in p},
        "output": "bivariate Gaussian (mean correction + Cholesky factor) per bus",
        "pooling": p["agg"], "chi2_fusion": p["fuse"]},
    "training": {"optimizer": "AdamW", "lr": p["lr"], "weight_decay": p["wd"], "scheduler": "cosine annealing", "batch_size": 128,
                 "epochs": p["epochs"], "early_stopping_patience": p["patience"], "early_stopping_metric": "validation-normal NLL",
                 "grad_clip": 1.0, "loss": "Gaussian NLL over all buses (Eq. 6)", "class_balancing": "n/a (normal data only)", "augmentation": "none"},
    "data": {"systems": ["ieee14", "ieee39", "ieee118"], "protocols": {"P1": "paper-faithful (datagen.py)", "P2": "realistic (datagen_p2.py)"},
             "split": "P1 random 7000/1500/1500; P2 temporal week split 70/15/15 % (7000/1500/1500 snapshots)",
             "replicates": [0, 1, 2, 3, 4], "features": "full measurement vector z (m channels) -> state via SE",
             "preprocessing": "all statistics (state means/stds, precision) fit on TRAIN normals only"},
    "threshold": {"rule": "split-conformal quantile of validation-normal scores", "alpha_primary": 0.05, "alpha_secondary": 0.01},
    "seeds": "replicate index r is used as data seed offset and torch/numpy seed",
    "model_selection_metric": "macro-F1 (val_mix) at alpha=0.05",
    "hyperparameters_passed_to_registry": p,
    "code_sha256": code_hash,
    "policy": "Frozen before any TEST evaluation of this model. No redesign based on TEST results.",
}
open(os.path.join(ROOT, "FINAL_MODEL_CONFIG.yaml"), "w").write(yaml.safe_dump(cfg, sort_keys=False))
os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
json.dump(p, open(os.path.join(ROOT, "configs", "final_hparams.json"), "w"), indent=1)
print(yaml.safe_dump(cfg, sort_keys=False))
