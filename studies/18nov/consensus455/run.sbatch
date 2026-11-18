#!/bin/bash
#SBATCH -A gehler
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 10
#SBATCH -J ni26-cap455
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
export PYTHONPATH=$base/ni26_history443_20261118:$base/ni26_wave379_v1_20261118:$base/ni26_fusion383_v1_20261118
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$base/neural-interfaces-2026/.venv/bin/python
sha256sum -c --status source.sha256
for parent in ${PYTHONPATH//:/ }; do
    (cd "$parent" && sha256sum -c --status source.sha256)
done
if [[ $1 == smoke ]]; then
    (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
fi
grep -Fx OK replay.log
"$py" run.py "$1"
