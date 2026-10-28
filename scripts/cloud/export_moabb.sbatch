#!/usr/bin/env bash
#SBATCH --job-name=ni26-moabb
#SBATCH --partition=cpu-galvani
#SBATCH --account=gehler
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
# Usage: sbatch --array=0-N scripts/cloud/export_moabb.sbatch DATASET N_SUBJECTS CHUNK
# Task i windows subjects i*CHUNK+1 .. min((i+1)*CHUNK, N_SUBJECTS) into
# data/<dataset>/windows_<lo>-<hi>.npz (scripts/export_moabb.py).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" MNE_DATA="$PWD/data/moabb_raw"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$MNE_DATA"
LO=$(( SLURM_ARRAY_TASK_ID * $3 + 1 ))
HI=$(( LO + $3 - 1 < $2 ? LO + $3 - 1 : $2 ))
python -I -W ignore scripts/export_moabb.py "$1" --subjects "$LO-$HI"
