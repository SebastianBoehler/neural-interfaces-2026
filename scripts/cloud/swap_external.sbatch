#!/usr/bin/env bash
#SBATCH --job-name=ni26-swap
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Usage: sbatch [--qos=vs] scripts/cloud/swap_external.sbatch SRC OUT EXT TRAIN
# Retrains raw ShallowConvNet + FBCNet slots of package SRC with external
# data EXT (scripts/swap_external.py, 100% harness agreement asserted), then
# replays OUT through the official ingestion + scoring on this node's CPU.
# Nothing is uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f runs/cloud/PREPARED
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/swap_external.py "$1" "$2" --ext "$3" --train "$4"
rm -rf "${2}_replay_linuxcpu"
python scripts/replay_package.py "$2" "${2}_replay_linuxcpu" cpu
