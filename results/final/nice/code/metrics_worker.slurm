#!/bin/bash
#SBATCH --job-name=MetricsSel
#SBATCH -A proj_1716
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --output=../assets/logs/metrics_sel_%j.log

module purge
module load Python/Miniconda_v25
source activate nk_env_clean
PYTHON_PATH="/home/ikvasilev/.conda/envs/nk_env_clean/bin/python"
export PYTHONUNBUFFERED=1

# Запуск расчета и отбора метрик
$PYTHON_PATH run_metric_selection.py