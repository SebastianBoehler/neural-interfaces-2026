#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,cpu-galvani
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 10
#SBATCH -J ni26-friday-c1
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
export PYTHONPATH=$base/ni26_wave379_v1_20261118:$base/ni26_fusion383_v1_20261118:$base/ni26_soft_csp361_v1_20261117
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$base/neural-interfaces-2026/.venv/bin/python
sha256sum -c --status source.sha256
run=$PWD
(cd "$base" && sha256sum -c --status "$run/parents.sha256")
if [[ $1 == smoke ]]; then (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log; fi
grep -Fx OK replay.log
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
case "$1" in
  smoke)
    "$py" produce.py within0 new smoke
    "$py" produce.py within0 returning smoke
    ;;
  produce)
    order="${orders[$SLURM_ARRAY_TASK_ID]}"
    "$py" produce.py "$order" new
    "$py" produce.py "$order" returning
    ;;
  score) "$py" collect.py ;;
  *) exit 2 ;;
esac
