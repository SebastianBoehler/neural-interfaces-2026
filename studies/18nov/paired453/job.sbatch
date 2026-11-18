#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani,v100-galvani
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 4
#SBATCH -J ni26-score453
set -euo pipefail
sha256sum --quiet -c sources.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python collect.py
