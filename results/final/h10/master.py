# coding: utf-8
import os
import numpy as np
from candidate_scorer import candidate_score
from metrics import TimeSeries, chamfer_distance_metric

TARGET_R = 28.0
SERIES_SIZE = 10000

print("--- STEP 1: P-Frob Selection (10,000 series) ---")
np.random.seed(42)
r_candidates = np.random.uniform(23.0, 33.0, 10000)

pf_scores = []
for i, r in enumerate(r_candidates):
    # Используем твою функцию, которая внутри создает TimeSeries
    dist = candidate_score(TARGET_R, SERIES_SIZE, r, SERIES_SIZE, n_bins=32)
    pf_scores.append((r, dist))
    if i % 1000 == 0:
        print("Processed {0}/10000".format(i))

pf_scores.sort(key=lambda x: x[1])
pf_selected_rs = [x[0] for x in pf_scores[:500]]

print("--- STEP 2: Chamfer Selection (500 series) ---")
target_ts_values = TimeSeries(series_type="Lorentz", size=SERIES_SIZE, r=TARGET_R).values
chamfer_scores = []
for r in pf_selected_rs:
    ts_cand_values = TimeSeries(series_type="Lorentz", size=SERIES_SIZE, r=r).values
    dist = chamfer_distance_metric(target_ts_values, ts_cand_values)
    chamfer_scores.append((r, dist))

chamfer_scores.sort(key=lambda x: x[1])
best_donors = [x[0] for x in chamfer_scores[:50]]

print("--- STEP 3: Random Selection ---")
np.random.seed(123)
remaining_rs = list(set(r_candidates) - set(best_donors))
random_donors = np.random.choice(remaining_rs, 50, replace=False)

print("--- STEP 4: Submitting 102 jobs to Slurm ---")

# Бейзлайны
os.system("sbatch worker.slurm BASELINE_10K 28.0")
os.system("sbatch worker.slurm BASELINE_30K 28.0")

# Лучшие
for r in best_donors:
    os.system("sbatch -A proj_1716 worker.slurm BEST {0}".format(r))

# Рандомные
for r in random_donors:
    os.system("sbatch -A proj_1716 worker.slurm RAND {0}".format(r))

print("All tasks submitted. Use 'mj' or 'squeue -u ikvasilev' to monitor.")
