#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-preemptable-galvani
#SBATCH --nodelist=galvani-cn102
#SBATCH --gres=none
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 6
#SBATCH -J ni26-ext418
set -eu
export NI26_DEVICE=cpu OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
py=../neural-interfaces-2026/.venv/bin/python
sha256sum -c --status source.sha256
if [[ "$1" == score ]]; then
    (cd ../ni26_s318_coral_fbl_20261117 && ../neural-interfaces-2026/.venv/bin/python replay/check.py) > replay.txt
    grep -x OK replay.txt
    "$py" score.py
else
    "$py" worker.py "${SLURM_ARRAY_TASK_ID:-1}"
fi
