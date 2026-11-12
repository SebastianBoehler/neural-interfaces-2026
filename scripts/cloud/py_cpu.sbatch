#!/usr/bin/env bash
#SBATCH --job-name=ni26-cpu
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=02:00:00
# Usage: sbatch scripts/cloud/py_cpu.sbatch SCRIPT.py [ARGS...]   (CPU-only study script)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python "$@"
