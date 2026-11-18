#!/bin/bash
#SBATCH -A gehler
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 8
#SBATCH -J ni26-hist443
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
wave=$base/ni26_wave379_v1_20261118
fusion=$base/ni26_fusion383_v1_20261118
export PYTHONPATH=$wave:$fusion OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$base/neural-interfaces-2026/.venv/bin/python
sha256sum -c --status source.sha256
(cd "$wave" && sha256sum -c --status source.sha256)
(cd "$fusion" && sha256sum -c --status source.sha256)
case "$1" in
    smoke) (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
           grep -Fx OK replay.log
           "$py" actor.py 379 P01
           "$py" score.py smoke ;;
    actor) printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"
           "$py" actor.py "$2" "$person" ;;
    score) grep -Fx OK replay.log
           "$py" score.py "$2" ;;
esac
