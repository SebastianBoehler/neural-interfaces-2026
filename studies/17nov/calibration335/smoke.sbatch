#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s335-smoke
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export CAL_ROOT="$PWD" CAL_SMOKE=1 FIXED_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_gen334_v4_20261117/fixed_features OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B study.py
