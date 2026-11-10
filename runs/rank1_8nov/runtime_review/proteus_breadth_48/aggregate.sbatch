#!/usr/bin/env bash
#SBATCH --job-name=ni26-proteus-breadth-score
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --output=runs/rank1_8nov/runtime_review/proteus_breadth_48/aggregate-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
wave=runs/rank1_8nov/runtime_review/proteus_breadth_48
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum -c "$wave/inputs.sha256"
.venv/bin/python -u "$wave/aggregate.py" "$wave" "$1" "$wave/summary-$1.json"
