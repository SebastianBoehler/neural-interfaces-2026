#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-c21g
#SBATCH --output=c21g-%j.out
set -euo pipefail
root=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c source.sha256
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "${1:?select or score}" in
    select) exec "$root/.venv/bin/python" -B select_fbl2.py ;;
    score) exec "$root/.venv/bin/python" -B score_gate.py ;;
    *) exit 2 ;;
esac
