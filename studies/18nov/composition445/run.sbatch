#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 10
#SBATCH -J ni26-composition445
set -euo pipefail
export NI26_DEVICE=cpu OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/lustre/work/gehler/gfh527/ni26_onebn381_actor_v1_20261118
py=../neural-interfaces-2026/.venv/bin/python
sha256sum -c --status source.sha256
case "$1" in
    smoke) "$py" actor.py lock; "$py" actor.py P01 ;;
    held) people=(P02 P03 P09); "$py" actor.py "${people[$SLURM_ARRAY_TASK_ID]}" ;;
    score) (cd ../ni26_s318_coral_fbl_20261117 && ../neural-interfaces-2026/.venv/bin/python replay/check.py) > replay.log
           grep -Fx OK replay.log
           "$py" score.py ;;
    *) exit 2 ;;
esac
