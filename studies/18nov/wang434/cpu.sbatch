#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,v100-galvani,a100-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 12
#SBATCH -J ni26-wang434
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 NI26_DEVICE=cpu
py=../neural-interfaces-2026/.venv/bin/python
sha256sum -c --status source.sha256
case "$1" in
    prepare) "$py" prepare434.py "$SLURM_ARRAY_TASK_ID" ;;
    a6smoke) "$py" a6.py 8 smoke ;;
    a6) "$py" a6.py "$SLURM_ARRAY_TASK_ID" ;;
    score) (cd ../ni26_s318_coral_fbl_20261117 && ../neural-interfaces-2026/.venv/bin/python replay/check.py) > replay.txt
           grep -x OK replay.txt && "$py" score434.py ;;
    *) exit 2 ;;
esac
