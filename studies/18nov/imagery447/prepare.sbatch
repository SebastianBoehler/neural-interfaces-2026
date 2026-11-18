#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 12
#SBATCH -J ni26-chang447p
set -eu
repo=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
people=(211 216 221 222 226 227)
sha256sum -c --status sources.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$repo/.venv/bin/python" prepare.py "$repo" "${people[${SLURM_ARRAY_TASK_ID:-0}]}"
