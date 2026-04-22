#!/bin/bash
#SBATCH --job-name=MasterSelect
#SBATCH -A proj_1716
#SBATCH --output=/home/ikvasilev/PaTHoP/assets/logs/fca_%j.out
#SBATCH --error=/home/ikvasilev/PaTHoP/assets/logs/fca_%j.out
#SBATCH --time=120:30:00

module purge
module load Python/Miniconda_v25
source activate nk_env_clean
PYTHON_PATH="/home/ikvasilev/.conda/envs/nk_env_clean/bin/python"
export PYTHONUNBUFFERED=1

# $1 и $2 — это Группа (BEST/RAND) и значение R, которые придут из мастер-скрипта
$PYTHON_PATH master.py
