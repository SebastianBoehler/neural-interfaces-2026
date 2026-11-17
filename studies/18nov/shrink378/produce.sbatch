#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-shrink378
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if [[ ${1:-} == smoke ]]; then
    srun --cpu-bind=cores "$py" -B produce.py P01 smoke
else
    printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID + 1))"
    srun --cpu-bind=cores "$py" -B produce.py "$person"
fi
