#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-galvani,2080-preemptable-galvani,v100-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 12
#SBATCH -J ni26-panel410
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
i=${SLURM_ARRAY_TASK_ID:-0}
if [[ $1 == smoke ]]; then
    (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) >replay.log
    grep -Fx OK replay.log
fi
if [[ $1 == order ]]; then
    "$py" panel.py order "$((i/8))" "$((i%8))"
elif [[ $1 == score ]]; then
    "$py" score.py
else
    "$py" panel.py "$1" "$i" 0
fi
