#!/bin/bash
#SBATCH --job-name=ni26-js321-train
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:06:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" train.py "$SLURM_ARRAY_TASK_ID"
