# PALOC: Physics-Anchored Leave-One-Bus-Out Conditional Likelihood

PALOC detects and localizes stealthy AC false data injection attacks (FDIAs) using only normal operating data. The pipeline has four steps:
- **Estimation:** a fixed-gain (chord) AC state estimator maps each measurement snapshot to a state estimate. Missing telemetry channels get zero weight.
- **Anchor:** a closed-form linear-Gaussian leave-one-bus-out model gives the conditional innovation of every bus.
- **Blind-spot network:** this network excludes the tested bus from its own context by construction. It refines each conditional with a heteroscedastic 2×2 Gaussian.
- **Decision:** the maximum of the learned bus statistics, calibrated by a split-conformal threshold, raises the alarm. The arg-max of the anchor statistics gives the attacked bus.

This repository contains the complete source code. It covers the method, the data generation, all benchmarks, the hyper-parameter search, the evaluation and the statistical analysis. With it, every experiment of the accompanying paper can be reproduced on the IEEE 14-, 39- and 118-bus systems.

## Repository structure

- `src/physics.py`: AC measurement function in PyTorch; fixed-gain chord estimator with missing-channel support; WLS.
- `src/grid.py`: network models and admittance matrices built with pandapower.
- `src/candidates.py`: PALOC core (blind-spot conditional network, linear anchor, training and scoring).
- `src/paloc2.py`: the final PALOC configuration (detection and localization read-outs) and its ablation variants.
- `src/datagen.py`, `src/datagen_p2.py`, `src/profiles.py`: normal operation and attack generation for protocols P1 and P2.
- `src/evalsets_v2.py`, `src/robust_sets.py`: detector-aware, load-redistribution, noise, missing-channel and line-outage evaluation sets.
- `src/registry.py`, `src/ae_models.py`, `src/bench_v2.py`, `src/supervised.py`: all benchmark detectors.
  - Model-based: chi-square bad-data detection, Bayesian GLRT, Mahalanobis test on the state estimate.
  - Classical: PCA, isolation forest, one-class SVM, kNN.
  - Deep autoencoders and one-class models: SAE, VAE, graph autoencoder, physics-informed autoencoder, Deep SVDD, PI-BAE.
  - Supervised references.
- `src/run.py`, `src/run_v2.py`: experiment drivers. Each run writes one metric file per (method, system, replicate).
- `src/tune_candidates.py`, `src/tune_deep.py`, `src/tune_benchmarks.py`, `src/select_v2.py`: hyper-parameter search and model selection on validation data.
- `src/loc_v2.py`, `src/localization.py`, `src/lowdata.py`, `src/impact_analysis.py`, `src/efficiency.py`, `src/se_check.py`: localization, training-data efficiency, attack impact, computational cost and estimator accuracy.
- `src/aggregate_v2.py`, `src/aggregate.py`: paired statistics (confidence intervals, effect sizes, Holm-corrected tests).
- `src/freeze_config.py`, `src/make_manifest.py`: configuration freezing and data-split manifest.
- `configs/`: the frozen hyper-parameters used in the paper.
  - `final_hparams.json`: PALOC.
  - `deep_hp.json`: the tuned deep baselines.
  - `benchmark_hp.json`: the classical detectors.
- `scripts/run_all.sh`: end-to-end reproduction of the experiments.

## Installation

The code was developed with Python 3.13 on Windows 11 and an NVIDIA GPU (CUDA 12.8). It also runs on Linux, and on the CPU only, at a slower speed.

```
git clone https://github.com/Mmassaoudi-cell/PALOC.git
cd PALOC
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the reported results. The PyTorch line refers to the nightly CUDA 12.8 build used in the experiments. A stable PyTorch release (2.6 or newer) for your platform gives equivalent results; install it from https://pytorch.org first, then the remaining packages. Set `RUN_DEVICE=cpu` to force CPU execution.

## Data

The test systems ship with pandapower, and protocol P1 needs no external data. Protocol P2 uses real load and photovoltaic profiles. These are not redistributed here, so download them from the original providers:

- **Ausgrid Solar Home Electricity Data, 1 July 2012 to 30 June 2013** (300 customers, half-hourly): https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data. Extract `2012-2013 Solar home electricity data v2.csv`.
- **ISO New England hourly system load** (used only for the transmission-level sensitivity study): https://www.iso-ne.com/isoexpress/web/reports/load-and-demand. Save the hourly system load as `isone_load.csv`, semicolon-separated, with the columns `date` (dd/mm/yyyy), `hour` (1–24) and `load`.

Place both files in `raw_data/` at the repository root, or point the environment variable `PALOC_RAW_DATA` to the folder that contains them.

All generated data are written to `data/` and all results to `results/`. This covers the snapshots, attacks, evaluation sets and profile cache. Both folders are created automatically and are not tracked by git.

## Generating the data

Protocol P1 (independent load scaling, random split):

```
cd src
python datagen.py --system ieee14 --reps 0 1 2 3 4
python datagen.py --system ieee39 --reps 0 1 2 3 4
python datagen.py --system ieee118 --reps 0 1 2 3 4
```

Protocol P2 (real load and PV profiles, reactive-power limits, distributed slack, temporal split):

```
python profiles.py
python datagen_p2.py --system ieee14 --reps 0 1 2 3 4 5 6 7 8 9
python datagen_p2.py --system ieee39 --reps 0 1 2 3 4 5 6 7 8 9
python datagen_p2.py --system ieee118 --reps 0 1 2 3 4 5 6 7 8 9
```

The first run of an experiment on a dataset builds its evaluation sets automatically and caches them in `data/`. These sets cover detector-aware attacks, load-redistribution attacks, noise scaling, missing channels and N−1 outages.

## Training and evaluating PALOC

Each call trains the detector on the normal training snapshots of one replicate. It then calibrates the conformal threshold on validation normals and evaluates all test sets:

```
python run_v2.py --methods PALOC2 --systems ieee14p2 ieee39p2 ieee118p2 --reps 0 1 2 3 4 5 6 7 8 9 --tag v2
```

The metrics are written to `results/raw/v2/` and the raw scores to `results/scores/v2/`. They include AUC, TPR, FPR, macro-F1, balanced accuracy and PR-AUC at three false-alarm budgets, with nominal and recalibrated thresholds.

Bus localization:

```
python loc_v2.py --methods PALOC2 BayesGLRT PIBAE-t SAE-t
```

## Reproducing the paper's experiments

`scripts/run_all.sh` runs the complete pipeline in order. The individual steps are:

1. **Data generation:** see above. For the sensitivity study, generate the variants with `datagen_p2.py --kappa 0.2|0.5 --pv 0|0.4 --source ausgrid|isone --suffix k02|k05|pv0|pv40|iso`.
2. **Hyper-parameter search (validation data only):** run `python tune_deep.py --model <SAE|VAE|GraphAE|PIAE|PIBAE|DeepSVDD> --trials 12` for the deep baselines. The frozen results of all searches are already provided in `configs/`, so this step is optional.
3. **Main benchmark:**
   - run PALOC2 and the thirteen unsupervised detectors with `run_v2.py` on P2 (replicates 0–9) and P1 (replicates 0–4): BDD, PCA, IForest, OCSVM, kNN, StateMaha, BayesGLRT, SAE-t, VAE-t, GraphAE-t, PIAE-t, DeepSVDD-t and PIBAE-t;
   - run the supervised references on the physics-state features (SupS-XGB, SupS-LGBM, SupS-MLP) on replicates 0–4;
   - run the raw-measurement supervised references with `run.py --tag main`.
4. **Ablation:** `run_v2.py --methods PALOC2-noNN PALOC2-noAnchor PALOC2-diag PALOC2-sum PALOC2-learnedSE PALOC2-fullWLS PALOC2-labelfree --tag v2abl`.
5. **Sensitivity:** `run_v2.py` on the P2 variants with `--tag sens`.
6. **Localization, impact, training-data efficiency and cost:**
   - `loc_v2.py` and `impact_analysis.py`;
   - `lowdata.py --methods PALOC PALOC-noNN StateMaha PIBAE SAE`;
   - with `RUN_DEVICE=cpu OMP_NUM_THREADS=1`: `efficiency.py --methods <detectors> --systems ieee14p2 ieee39p2 ieee118p2 --out efficiency_v2.csv`.
7. **Statistics:** `python aggregate_v2.py`. It computes per-system paired differences with 95% confidence intervals, paired effect sizes, Holm-corrected paired t-tests and Wilcoxon signed-rank tests for every benchmark and scenario family.

`src/jobs_v2.sh <jobfile> <parallel>` runs a list of `run_v2.py` argument lines in parallel. Every run skips results that already exist, so interrupted experiments can be resumed.

## Reproducibility notes

- Every replicate regenerates the load scenarios, measurement noise, data splits, attacks and network initialization from its replicate seed.
- All model and threshold choices use training and validation data only. The final configuration was frozen before test evaluation; `freeze_config.py` records the code hashes.
- The conformal threshold guarantees a marginal false-positive rate of at most the chosen budget on exchangeable normal data. Re-estimate it from recent normal snapshots whenever the noise level, measurement configuration or topology changes.

## Citation

If you use this code, please cite the accompanying paper:

```
@article{massaoudi2026paloc,
  author  = {Massaoudi, Mohamed and Ez Eddin, Maymouna},
  title   = {Physics-Anchored Leave-One-Bus-Out Conditional Likelihood for Unsupervised Detection and Localization of Stealthy {AC} False Data Injection Attacks},
  journal = {IEEE Transactions on Smart Grid},
  note    = {Submitted},
  year    = {2026}
}
```

## License

MIT License; see `LICENSE`.

## Contact

Mohamed Massaoudi, Texas A&M University and Tarleton State University: mohamed.massaoudi@tamu.edu
