#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-l1-parallel-smoke
#SBATCH --output=logs/parallel-smoke-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain
sha256sum -c chain-src.sha256
sha256sum -c parallel-src.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/lane1/parallel_smoke.py
