#!/bin/bash
#SBATCH --job-name=ni26-js321-shared-release
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python shared_release.py
