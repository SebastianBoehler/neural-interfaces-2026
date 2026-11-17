#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-lag364-guard
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c finish-source.sha256
sha256sum -c chain-source.sha256
(cd chain && sha256sum -c ../runtime.sha256 >/dev/null)
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
case ${1:?} in
    assemble) "$PY" -B assemble.py ;;
    score) "$PY" -B score.py ;;
    *) exit 2 ;;
esac
