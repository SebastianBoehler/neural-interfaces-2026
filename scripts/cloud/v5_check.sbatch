#!/usr/bin/env bash
#SBATCH --job-name=ni26-v5check
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani,a100-preemptable-galvani,v100-galvani
#SBATCH --exclude=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
# Usage: sbatch scripts/cloud/v5_check.sbatch PKG_V5 PKG_BASE   (integration check only; CPU replays of both)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/check_own_stage.py "${1%/}" "${2%/}" --out "${1%/}_check.json"
