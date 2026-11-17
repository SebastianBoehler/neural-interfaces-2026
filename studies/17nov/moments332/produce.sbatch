#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ni26-s332-views
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export MOMENT_ROOT="$PWD" OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
TASK=${SLURM_ARRAY_TASK_ID:?}
HELD=$(printf 'P%02d' "$((TASK/2+1))")
SEED=$((TASK%2))
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B produce332.py "$HELD" "$SEED"
