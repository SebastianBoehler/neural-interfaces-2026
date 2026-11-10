#!/usr/bin/env bash
#SBATCH --job-name=ni26-proteus-mechanisms
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=3
#SBATCH --mem=9G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=runs/rank1_8nov/runtime_review/proteus_mechanism_lanes/slurm-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
review=runs/rank1_8nov/runtime_review
wave="$review/proteus_mechanism_lanes"
lanes=(copula nuisance time)
lane=${lanes[$SLURM_ARRAY_TASK_ID]}
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
sha256sum -c "$wave/inputs.sha256"
.venv/bin/python -u "$wave/lane.py" "$lane" "$review/actual_runner_wave/artifacts" \
    "$review/actual_seed_wave/seed100" "$review/actual_seed_wave/seed200" \
    "$review/proteus_covariance_overlap_wave/fit-3081051.json" "$wave/$lane-$SLURM_ARRAY_JOB_ID.json"
