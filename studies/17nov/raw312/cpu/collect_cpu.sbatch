#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s312-cpuscore
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export NI26_WS="$PWD" NI26_ROOT="$PWD" NI26_CACHE=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_data1/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -B collect.py
