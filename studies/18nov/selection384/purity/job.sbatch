#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-galvani,2080-preemptable-galvani,v100-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-purity384
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B check384.py "$1"
