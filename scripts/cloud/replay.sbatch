#!/usr/bin/env bash
#SBATCH --job-name=ni26-replay
#SBATCH --partition=a100-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Usage: sbatch scripts/cloud/replay.sbatch runs/submissions/PKG [OUT_DIR]
# Official ingestion + scoring replay of one package on an A100 node with
# COMPET_DEVICE=cuda (as on the worker); writes OUT_DIR/replay.json with the
# ingestion wall time. Nothing is uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f runs/cloud/PREPARED
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset NEURALBENCH_EXTRACT_CPUS
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'"
nvidia-smi --query-gpu=name,driver_version --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"
uptime
python scripts/replay_package.py "$1" "${2:-$1_replay_a100_$SLURM_JOB_ID}" cuda
