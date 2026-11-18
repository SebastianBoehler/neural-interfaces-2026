#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani,v100-galvani
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 12
#SBATCH -J ni26-cue425p
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
../neural-interfaces-2026/.venv/bin/python crops.py "${SLURM_ARRAY_TASK_ID:-$1}"
