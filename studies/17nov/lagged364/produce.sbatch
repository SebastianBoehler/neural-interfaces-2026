#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-lag364-fold
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B relation.py
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then printf -v holdout 'P%02d' "$SLURM_ARRAY_TASK_ID"; else holdout=${1:?}; fi
"$PY" -B producer.py "$holdout"
