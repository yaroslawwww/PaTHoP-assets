#!/bin/bash
#SBATCH --job-name=ev_task
#SBATCH -A proj_1716
#SBATCH -c 1
#SBATCH --time=24:00:00
#SBATCH --output=../assets/logs/worker_%j.log

module purge
module load Python/Miniconda_v25
source activate nk_env_clean
PYTHON_PATH="/home/ikvasilev/.conda/envs/nk_env_clean/bin/python"

# $1 - Название группы (например, EXP1_CLEAR)
# $2 - Параметр (Размер N для первого эксперимента)
# $3 - Горизонт h (1 или 20)
$PYTHON_PATH eval_worker.py $1 $2 $3