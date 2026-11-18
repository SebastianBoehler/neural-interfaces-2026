#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 5
#SBATCH -J ni26-friday-d1
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$base/ni26_qualify385_v1_20261118:$repo/experiments
sha256sum --quiet -c d1.sha256
printf '%s  %s\n' 390fd6bea0ddc9c53e34fdabb094b1a269ff45b07b4a5542a8135b44d5314a2c "$repo/experiments/protocol.py" | sha256sum --quiet -c -
(
    cd "$base/ni26_s318_coral_fbl_20261117"
    "$repo/.venv/bin/python" replay/check.py
) > replay.log
grep -Fx OK replay.log
"$repo/.venv/bin/python" d1.py
