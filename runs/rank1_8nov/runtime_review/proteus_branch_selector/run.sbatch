#!/usr/bin/env bash
#SBATCH --job-name=ni26-branch-selector
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=runs/rank1_8nov/runtime_review/proteus_branch_selector/slurm-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
review=runs/rank1_8nov/runtime_review
wave="$review/proteus_branch_selector"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
sha256sum -c "$wave/inputs.sha256"
.venv/bin/python -u "$wave/selector.py" "$review/actual_runner_wave/artifacts" \
    "$review/actual_seed_wave/seed100" "$review/actual_seed_wave/seed200" \
    "$review/proteus_covariance_overlap_wave/fit-3081051.json" "$wave/result-$SLURM_JOB_ID.json"
