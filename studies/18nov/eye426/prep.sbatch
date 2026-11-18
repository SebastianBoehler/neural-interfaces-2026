#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 10
#SBATCH -J ni26-eye426p
set -eu
sha256sum -c --status sources.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
../neural-interfaces-2026/.venv/bin/python prepare.py "${SLURM_ARRAY_TASK_ID:-1}" "$@"
