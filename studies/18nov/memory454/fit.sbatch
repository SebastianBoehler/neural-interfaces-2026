#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 6
#SBATCH -J ni26-mem454
set -euo pipefail
sha256sum --quiet -c source.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=../venv/bin/python
if [[ $1 == smoke ]]; then
    "$py" fit.py kernel 454 P01 smoke
    "$py" fit.py covariance 454 P01 smoke
else
    printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID%10+1))"
    "$py" fit.py "$1" "$((454+1000*(SLURM_ARRAY_TASK_ID/10)))" "$person"
fi
