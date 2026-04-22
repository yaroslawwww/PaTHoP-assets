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
# $1 и $2 — это Группа (BEST/RAND) и значение R, которые придут из мастер-скрипта
$PYTHON_PATH eval_worker.py $1 $2
