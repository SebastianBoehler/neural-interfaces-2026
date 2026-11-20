#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-laneK2-feat
#SBATCH --output=laneK2-%x-%j.out
# usage (cd to the lane K2 chain dir): sbatch feat.sbatch
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec $base/neural-interfaces-2026/.venv/bin/python -B feat_k2.py $base/ni26_s318_coral_fbl_20261117/data/proteus_windows.npz feat_k2.npz
