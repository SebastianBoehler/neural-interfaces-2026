#!/usr/bin/env bash
#SBATCH --job-name=ni26-sweep
#SBATCH --partition=a100-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
# Usage: sbatch --array=0-2 scripts/cloud/sweep.sbatch "cfg1 cfg2" "dev_loso,dev_session"
# The array index is the training seed. Results append to
# runs/experiments/results-<jobid>-<seed>.jsonl; nothing is uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f runs/cloud/PREPARED
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset NEURALBENCH_EXTRACT_CPUS
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'"
# Export once; concurrent array tasks wait on the lock.
exec 9>data/.export.lock
flock 9
test -f data/proteus_windows.npz || python scripts/export_windows.py
flock -u 9
SEED="${SLURM_ARRAY_TASK_ID:-0}"
export NI26_RESULTS="runs/experiments/results-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SEED}.jsonl"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python experiments/run_cv.py $1 --protocols "${2:-dev_loso,dev_session}" --seed "$SEED"
