#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,v100-galvani,a100-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 5
#SBATCH -J ni26-cover397b
set -eu
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256 >/dev/null
case "$1" in
 smoke) "$py" null397.py smoke S ;;
 full) if [[ $SLURM_ARRAY_TASK_ID == 0 ]]; then k=S; else k=P; fi; "$py" null397.py full "$k" ;;
 score) "$py" null397.py score ;;
esac
