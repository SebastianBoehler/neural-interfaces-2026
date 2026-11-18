#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 28
#SBATCH -J ni26-match456
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
export PYTHONPATH=$base/ni26_history443_20261118:$base/ni26_wave379_v1_20261118:$base/ni26_fusion383_v1_20261118
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c --status source.sha256
if [[ ${2:-} == smoke ]]; then
    (cd "$base/ni26_s318_coral_fbl_20261117" && "$base/neural-interfaces-2026/.venv/bin/python" replay/check.py) > "$1.replay.log"
fi
grep -Fx OK "$1.replay.log"
"$base/neural-interfaces-2026/.venv/bin/python" score456.py "$@"
