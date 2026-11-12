#!/usr/bin/env bash
#SBATCH --job-name=ni26-virt
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=01:30:00
# Usage: sbatch scripts/cloud/virt.sbatch MODE PARAM [COPIES]   (ledger s47; CPU only)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
MODE="$1" PARAM="$2" COPIES="${3:-2}" python studies/12nov/virt.py
