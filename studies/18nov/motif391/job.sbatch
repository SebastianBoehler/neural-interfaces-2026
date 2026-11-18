#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,v100-galvani,a100-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 8
#SBATCH -J ni26-motif391
set -eu
base=/mnt/lustre/work/gehler/gfh527
py=$base/neural-interfaces-2026/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256 >/dev/null
case "$1" in
 smoke) "$py" prepare.py smoke ;;
 prepare) "$py" prepare.py full; sha256sum features.npz >features.sha256; "$py" fit.py P01 smoke ;;
 fit) sha256sum -c features.sha256; printf -v p 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"; "$py" fit.py "$p" ;;
 score) (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) >replay.log; grep -x OK replay.log; "$py" score.py ;;
 report) "$py" report.py ;;
esac
