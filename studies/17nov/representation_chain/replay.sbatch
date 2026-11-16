#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s315-replay
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c representation-source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I replay/check.py
