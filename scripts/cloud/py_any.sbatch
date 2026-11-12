#!/usr/bin/env bash
#SBATCH --job-name=ni26-py
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani,a100-preemptable-galvani,v100-galvani
#SBATCH --exclude=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=01:00:00
# Usage: sbatch scripts/cloud/py_any.sbatch SCRIPT.py [ARGS...]   (short CPU study on a GPU node when the CPU partition is full)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python "$@"
