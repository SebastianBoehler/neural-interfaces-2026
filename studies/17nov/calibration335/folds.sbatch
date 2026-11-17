#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --array=4-10%4
#SBATCH --job-name=ni26-s335-fold
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export CAL_ROOT="$PWD" CAL_HELD="$(printf 'P%02d' "$SLURM_ARRAY_TASK_ID")" CAL_SMOKE_REPORT=/mnt/lustre/work/gehler/gfh527/ni26_cal335_v1_20261117/smoke.json FIXED_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_gen334_v4_20261117/fixed_features OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B study.py
