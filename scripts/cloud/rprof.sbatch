#!/usr/bin/env bash
#SBATCH --job-name=rprof
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=00:40:00
#SBATCH --output=runs/diag/gcpu/rprof-%j.out
# s262 stage 0: profile one v6.5b refit on a 2080. Usage: sbatch scripts/cloud/rprof.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=4
nvidia-smi -L
python -I studies/14nov/refit_profile.py runs/sealed/zt65b/cand --device cuda --prefixes 190 378
