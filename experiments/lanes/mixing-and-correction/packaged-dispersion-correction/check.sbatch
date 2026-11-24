#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-laneP2-check
#SBATCH --output=logs/check-%j.out
# usage (release dir): sbatch laneP2/check.sbatch TEST ZIP REF_ZIP PERSON...   (TEST zero | disp | fbl2; PKG_P2_WINDOWS default 120)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B laneP2/pkg_check_p2.py "$@"
