#!/usr/bin/env bash
#SBATCH --job-name=ni26-pt
#SBATCH --partition=a100-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
# Usage: sbatch --array=0-2 scripts/cloud/pretrain.sbatch ARCH ALIGN TAG CORPORA \
#            "cfg1 cfg2" "dev_loso,dev_session" ["--epochs 30 --aug"]
# The array index is the seed: pretrain runs/pretrain/ARCH_ALIGN-TAG-sSEED.pt
# (experiments/pretrain.py, skipped if present), then fine-tune and evaluate
# the configs (run_cv.py *_ptTAG*). Probabilities: runs/experiments/probs_r10.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f runs/cloud/PREPARED
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NI26_PROBS=runs/experiments/probs_r10
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -f "runs/pretrain/$1_$2-$3-s$SEED.pt" ||
  python experiments/pretrain.py "$1" "$2" "$3" "$4" --seed "$SEED" ${7:-}
export NI26_RESULTS="runs/experiments/results-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SEED}.jsonl"
test -z "$5" || python experiments/run_cv.py $5 --protocols "${6:-dev_loso,dev_session}" --seed "$SEED"
