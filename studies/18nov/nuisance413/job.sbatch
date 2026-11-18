#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-galvani,2080-preemptable-galvani,v100-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 15
#SBATCH -J ni26-nuis413
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
i=${SLURM_ARRAY_TASK_ID:-0}
case "$1" in
 smoke) (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) > replay.log
        grep -Fx OK replay.log
        "$py" study.py smoke ;;
 order) "$py" study.py order "$((i/8))" "$((i%8))" ;;
 actor) "$py" study.py actor "$i" 0 ;;
 score) "$py" score.py ;;
 *) "$py" study.py "$1" "$i" ;;
esac
