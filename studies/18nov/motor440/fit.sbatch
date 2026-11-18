#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-fit440
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
sha256sum -c fit.sha256
cp -n fit.sha256 source.sha256
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if [[ $1 == smoke ]]; then
    "$py" -B fit.py P01 smoke
else
    printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"
    "$py" -B fit.py "$person"
fi
