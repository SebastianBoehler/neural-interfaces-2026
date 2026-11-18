#!/bin/bash
#SBATCH -A gehler
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 15
#SBATCH -J ni26-n452
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
export PYTHONPATH=$b/ni26_wave379_v1_20261118
export LD_LIBRARY_PATH="$b/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$b/neural-interfaces-2026/.venv/bin/python
sha256sum -c --status source.sha256
sha256sum -c --status library.sha256
(cd "$PYTHONPATH" && sha256sum -c --status source.sha256)
case "$1" in
 smoke) "$py" run.py 379 P01 smoke ;;
 actor) printf -v p 'P%02d' "$((1+SLURM_ARRAY_TASK_ID%10))"
        if (( SLURM_ARRAY_TASK_ID < 10 )); then seed=379; else seed=1441; fi
        "$py" run.py "$seed" "$p" ;;
 score) "$py" collect.py ;;
esac
