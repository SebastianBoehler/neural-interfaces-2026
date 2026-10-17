#!/usr/bin/env bash
#SBATCH --job-name=ni26-gpu-validate
#SBATCH --partition=a100-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f runs/cloud/PREPARED
export PATH="$PWD/.venv/bin:$PATH"
export BENCHOPT_DATA_HOME="$PWD/data"
export COMPET_DEVICE=cuda
export BENCHOPT_DEBUG=true
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset NEURALBENCH_EXTRACT_CPUS
python scripts/cloud/validate_gpu.py
