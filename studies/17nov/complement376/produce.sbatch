#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:12:00
#SBATCH --job-name=ni26-comp376
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if [[ ${1:-} == smoke ]]; then
    srun --cpu-bind=cores "$py" -B produce.py P01 smoke
else
    if [[ "$SLURM_ARRAY_TASK_ID" != 0 ]]; then test -f P01/proof.json; fi
    printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID + 1))"
    srun --cpu-bind=cores "$py" -B produce.py "$person"
fi
