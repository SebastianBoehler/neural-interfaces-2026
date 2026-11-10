#!/usr/bin/env bash
#SBATCH --job-name=ni26-spectral-score
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --no-requeue
#SBATCH --output=runs/rank1_8nov/runtime_review/proteus_multitaper_wave/aggregate-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
wave=runs/rank1_8nov/runtime_review/proteus_multitaper_wave
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum -c "$wave/inputs.sha256"
.venv/bin/python -u "$wave/spectral.py" aggregate "$wave"/seed-{0,100,200,300}-"$1".json \
    "$wave/summary-$SLURM_JOB_ID.json"
