#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 12
#SBATCH -J ni26-score454
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
wave=$base/ni26_wave379_v1_20261118
fusion=$base/ni26_fusion383_v1_20261118
export PYTHONPATH=$wave:$fusion OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$base/neural-interfaces-2026/.venv/bin/python
sha256sum --quiet -c source.sha256
(cd "$wave" && sha256sum --quiet -c source.sha256)
(cd "$fusion" && sha256sum --quiet -c source.sha256)
if [[ $1 == smoke ]]; then
    (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
    grep -Fx OK replay.log
    "$py" score.py kernel 454 smoke
else
    grep -Fx OK replay.log
    "$py" score.py "$1" "$2"
fi
