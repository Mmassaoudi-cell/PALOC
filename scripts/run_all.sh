#!/bin/bash
# End-to-end reproduction of the PALOC experiments.
# Usage: bash scripts/run_all.sh [parallel_jobs]   (raw Ausgrid / ISO-NE files in raw_data/ or $PALOC_RAW_DATA)
set -e
PAR=${1:-2}
cd "$(dirname "$0")/../src"
P2_REPS="0 1 2 3 4 5 6 7 8 9"
P1_REPS="0 1 2 3 4"
SUP_REPS="0 1 2 3 4"
UNSUP="BDD PCA IForest OCSVM kNN StateMaha BayesGLRT SAE-t VAE-t GraphAE-t PIAE-t DeepSVDD-t PIBAE-t"
SUPRAW="Sup-LR Sup-RF Sup-ET Sup-HGB Sup-XGB Sup-LGBM Sup-CatBoost Sup-MLP Sup-FTT Sup-GNN"
ABL="PALOC2-noNN PALOC2-labelfree PALOC2-sum PALOC2-diag PALOC2-noAnchor PALOC2-fullWLS PALOC2-learnedSE"

# 1. Data
for s in ieee14 ieee39 ieee118; do python datagen.py --system $s --reps $P1_REPS; done
python profiles.py
for s in ieee14 ieee39 ieee118; do python datagen_p2.py --system $s --reps $P2_REPS; done
for s in ieee14 ieee118; do
  python datagen_p2.py --system $s --reps 0 1 2 --kappa 0.2 --suffix k02
  python datagen_p2.py --system $s --reps 0 1 2 --kappa 0.5 --suffix k05
  python datagen_p2.py --system $s --reps 0 1 2 --pv 0 --suffix pv0
  python datagen_p2.py --system $s --reps 0 1 2 --pv 0.4 --suffix pv40
  python datagen_p2.py --system $s --reps 0 1 2 --source isone --suffix iso
done

# 2. (optional) equal-budget tuning of the deep baselines; frozen results are in configs/
# for m in SAE VAE GraphAE PIAE PIBAE DeepSVDD; do python tune_deep.py --model $m --trials 12; done

# 3-5. Main benchmark, supervised references, ablation and sensitivity (job list run in parallel)
JOBS=$(mktemp)
for r in $P2_REPS; do for s in ieee14p2 ieee39p2 ieee118p2; do
  echo "--methods PALOC2 $UNSUP --systems $s --reps $r --tag v2" >> $JOBS
  echo "--methods $ABL --systems $s --reps $r --tag v2abl" >> $JOBS
done; done
for r in $P1_REPS; do for s in ieee14 ieee39 ieee118; do
  echo "--methods PALOC2 $UNSUP --systems $s --reps $r --tag v2" >> $JOBS
done; done
for r in $SUP_REPS; do for s in ieee14 ieee39 ieee118 ieee14p2 ieee39p2 ieee118p2; do
  echo "--methods SupS-XGB SupS-LGBM SupS-MLP --systems $s --reps $r --tag v2" >> $JOBS
done; done
for v in k02 k05 pv0 pv40 iso; do
  for r in 0 1 2; do echo "--methods PALOC2 BayesGLRT PCA StateMaha SAE-t PIBAE-t --systems ieee14p2$v --reps $r --tag sens" >> $JOBS; done
  for r in 0 1; do echo "--methods PALOC2 BayesGLRT PCA StateMaha SAE-t PIBAE-t --systems ieee118p2$v --reps $r --tag sens" >> $JOBS; done
done
bash jobs_v2.sh $JOBS $PAR
python run.py --methods $SUPRAW --systems ieee14 ieee39 ieee118 ieee14p2 ieee39p2 ieee118p2 --reps $SUP_REPS --tag main

# 6. Localization, impact, training-data efficiency, estimator accuracy and computational cost
python loc_v2.py --methods PALOC2 BayesGLRT PIBAE-t SAE-t
python impact_analysis.py
python lowdata.py --methods PALOC PALOC-noNN StateMaha PIBAE SAE
python se_check.py
RUN_DEVICE=cpu OMP_NUM_THREADS=1 python efficiency.py --methods PALOC2 $UNSUP SupS-XGB SupS-LGBM \
  --systems ieee14p2 ieee39p2 ieee118p2 --out efficiency_v2.csv

# 7. Statistics
python aggregate_v2.py
