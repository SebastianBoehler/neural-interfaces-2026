#!/usr/bin/env bash
#SBATCH --job-name=ni26-ext
#SBATCH --partition=a100-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
# Usage: sbatch [--qos=vs] scripts/cloud/external.sbatch "cfg1@ext cfg2@ext" \
#            "dev_loso,dev_session" "0 1 2"
# One job runs every seed in turn (for single-job QOS slots); sweep.sbatch is
# the array variant. Results: runs/experiments/results-<jobid>-<seed>.jsonl,
# probabilities: runs/experiments/probs_r5/. Nothing is uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f runs/cloud/PREPARED
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NI26_PROBS=runs/experiments/probs_r5
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for SEED in ${3:-0}; do
  export NI26_RESULTS="runs/experiments/results-${SLURM_JOB_ID}-${SEED}.jsonl"
  python experiments/run_cv.py $1 --protocols "${2:-dev_loso,dev_session}" --seed "$SEED"
done
