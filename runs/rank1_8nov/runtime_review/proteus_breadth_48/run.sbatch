#!/usr/bin/env bash
#SBATCH --job-name=ni26-proteus-breadth
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --array=0-47%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:04:00
#SBATCH --no-requeue
#SBATCH --output=runs/rank1_8nov/runtime_review/proteus_breadth_48/slurm-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
review=runs/rank1_8nov/runtime_review
wave="$review/proteus_breadth_48"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum -c "$wave/inputs.sha256"
.venv/bin/python -u "$wave/breadth_run.py" "$SLURM_ARRAY_TASK_ID" "$review/actual_runner_wave/artifacts" \
    "$review/actual_seed_wave/seed100" "$review/actual_seed_wave/seed200" \
    "$review/proteus_covariance_overlap_wave/fit-3081051.json" "$wave" "$SLURM_ARRAY_JOB_ID"
