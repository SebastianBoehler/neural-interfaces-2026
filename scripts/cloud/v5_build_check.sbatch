#!/usr/bin/env bash
#SBATCH --job-name=ni26-v5build
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani,a100-preemptable-galvani,v100-galvani
#SBATCH --exclude=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
# Usage: sbatch scripts/cloud/v5_build_check.sbatch [BETA]
# Builds runs/sealed/cand_v5_own from runs/sealed/cand_v41_bio (own-space stage, ledger s45) and runs the
# integration check against the base package on all public windows (CPU replays of both packages).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=runs/sealed/cand_v5_own
python scripts/add_own_stage.py runs/sealed/cand_v41_bio "$OUT" beta="${1:-0.10}"
python scripts/check_own_stage.py "$OUT" runs/sealed/cand_v41_bio --out "${OUT}_check.json"
