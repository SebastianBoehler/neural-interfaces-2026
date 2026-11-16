#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lane2-%A_%a.out
#SBATCH --error=logs/lane2-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NI26_WS="$PWD"
task="${SLURM_ARRAY_TASK_ID:-${LANE2_TASK:?set LANE2_TASK for single job}}"
sha256sum -c inputs.sha256
exec /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/lane2/run.py "$task"
