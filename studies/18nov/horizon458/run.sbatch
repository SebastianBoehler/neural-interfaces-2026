#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 12
#SBATCH -J ni26-horizon458
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$b/ni26_match456_v2_20261118:$b/ni26_history443_20261118:$b/ni26_wave379_v1_20261118:$b/ni26_fusion383_v1_20261118:$b/ni26_kernel457_20261118/original"
py=$b/neural-interfaces-2026/.venv/bin/python
sha256sum --quiet -c source.sha256
(cd "$b/ni26_kernel457_20261118/original" && sha256sum --quiet -c actor_source.sha256)
case "$1" in
 smoke)
    (cd "$b/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
    grep -Fx OK replay.log
    "$py" actor.py P01 smoke ;;
 actor) grep -Fx OK replay.log
    printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"
    "$py" actor.py "$person" ;;
 score) grep -Fx OK replay.log
    "$py" collect.py ;;
esac
