#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 12
#SBATCH -J ni26-anchor444
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
export WAVE379=$base/ni26_wave379_v1_20261118
export PYTHONPATH=$WAVE379 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
py=$base/neural-interfaces-2026/.venv/bin/python

sha256sum -c --status source.sha256
(cd "$WAVE379" && sha256sum -c --status source.sha256)
case "$1" in
    smoke) for arm in 379 1441; do (cd "$arm" && "$py" ../anchor.py P01 "$arm" smoke); done ;;
    fit) arms=(379 1441)
         arm=${arms[SLURM_ARRAY_TASK_ID/10]}
         printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID%10+1))"
         cd "$arm"
         "$py" ../anchor.py "$person" "$arm" ;;
    score) (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
           grep -Fx OK replay.log
           "$py" score.py ;;
esac
