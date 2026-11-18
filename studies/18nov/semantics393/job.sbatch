#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,v100-galvani,a100-galvani
#SBATCH -c 2
#SBATCH --mem=5G
#SBATCH -t 5
#SBATCH -J ni26-semantics393
set -eu
base=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
py=$base/.venv/bin/python
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256 >/dev/null
(cd "$base" && sha256sum -c "$SLURM_SUBMIT_DIR/inputs.sha256") >/dev/null
case "$1" in
 smoke|full) "$py" produce.py "$1" ;;
 score) "$py" score.py ;;
esac
