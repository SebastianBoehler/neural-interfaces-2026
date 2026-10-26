#!/usr/bin/env bash
#SBATCH --job-name=ni26-tta
#SBATCH --partition=a100-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# Usage: sbatch --array=0-4 scripts/cloud/tta.sbatch "cfg1 cfg2" "protocols"
# Trains each config once per fold (array index = seed) and saves its
# probabilities with and without time-shift TTA (experiments/tta_eval.py).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f runs/cloud/PREPARED
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NI26_PROBS=runs/experiments/probs_tta
for c in $1; do
  python experiments/tta_eval.py "$c" "${2:-dev_loso,dev_session,all_loso}" "${SLURM_ARRAY_TASK_ID:-0}"
done
