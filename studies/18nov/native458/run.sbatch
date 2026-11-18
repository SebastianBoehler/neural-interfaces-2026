#!/bin/bash
#SBATCH -A gehler
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 20
#SBATCH -J ni26-native458
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$b/ni26_match456_v2_20261118:$b/ni26_history443_20261118:$b/ni26_wave379_v1_20261118:$b/ni26_fusion383_v1_20261118:$b/ni26_kernel457_20261118/original"
py=$b/neural-interfaces-2026/.venv/bin/python
sha256sum --quiet -c source.sha256
case $1 in
 build) bash build.sh ;;
 smoke|bench)
    sha256sum --quiet -c library.sha256
    people=(P04 P10)
    if [[ $1 == smoke ]]; then
        "$py" bench.py P04 smoke
    else
        "$py" bench.py "${people[$SLURM_ARRAY_TASK_ID]}"
    fi ;;
 score_smoke) (cd "$b/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
    grep -Fx OK replay.log
    "$py" score_pair.py smoke ;;
 score) grep -Fx OK replay.log
    "$py" score_pair.py ;;
esac
