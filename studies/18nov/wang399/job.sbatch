#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,v100-galvani,a100-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 8
#SBATCH -J ni26-wang399
set -eu
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256 >/dev/null
case "$1" in
 smoke) "$py" fetch.py 2 smoke ;;
 fetch) "$py" fetch.py "$((SLURM_ARRAY_TASK_ID+2))" ;;
 produce) if [[ $SLURM_ARRAY_TASK_ID == 0 ]]; then target=P; else target=S; fi; "$py" produce.py "$target" ;;
 score|reserve) "$py" score.py "$1" ;;
esac
