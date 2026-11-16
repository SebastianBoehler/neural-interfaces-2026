#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-neutral-replay
#SBATCH --output=logs/neutral-replay-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s302_neutral_20261116/neutral
sha256sum -c chain-src.sha256
sha256sum -c neutral-src.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I replay/check.py
