#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,v100-galvani,a100-galvani
#SBATCH --mem=6G
#SBATCH -c 2
#SBATCH -t 6
#SBATCH -J ni26-cover397
set -eu
base=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
py=$base/.venv/bin/python
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256 >/dev/null
id=${SLURM_ARRAY_TASK_ID:-0}
case "$1" in
 smoke) (cd "$base" && sha256sum -c "$SLURM_SUBMIT_DIR/inputs.sha256") >/dev/null; "$py" smoke.py ;;
 prep) case "$id" in 0) "$py" direct.py prepare S ;; 1) "$py" direct.py prepare P ;; 2) "$py" shin.py S ;; 3) "$py" shin.py P ;; esac ;;
 fit) if ((id<10)); then "$py" direct.py fit S "$((id-1))"; else "$py" direct.py fit P "$((id-11))"; fi ;;
 score) "$py" score.py ;;
esac
