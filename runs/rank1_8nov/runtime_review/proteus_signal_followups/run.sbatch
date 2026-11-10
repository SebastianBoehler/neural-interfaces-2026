#!/usr/bin/env bash
#SBATCH --job-name=ni26-source-signal
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:08:00
#SBATCH --no-requeue
#SBATCH --output=runs/rank1_8nov/runtime_review/proteus_signal_followups/slurm-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
review=runs/rank1_8nov/runtime_review
wave="$review/proteus_signal_followups"
screens=(geometry direction)
screen="${screens[$SLURM_ARRAY_TASK_ID]}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
sha256sum -c "$wave/inputs.sha256"
.venv/bin/python -u "$wave/run_screen.py" "$screen" "$review/actual_runner_wave/artifacts" \
    "$review/actual_seed_wave/seed100" "$review/actual_seed_wave/seed200" \
    "$review/proteus_covariance_overlap_wave/fit-3081051.json" \
    "$wave/$screen-$SLURM_ARRAY_JOB_ID.json"
