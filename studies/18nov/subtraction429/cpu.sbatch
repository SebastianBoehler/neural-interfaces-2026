#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,v100-galvani,a100-galvani,2080-preemptable-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 8
#SBATCH -J ni26-wang429
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 NI26_DEVICE=cpu
base=/mnt/lustre/work/gehler/gfh527
py=$base/neural-interfaces-2026/.venv/bin/python
(cd "$base/ni26_wang399_v2_20261118" && sha256sum -c --status source_ranges.sha256)
sha256sum -c --status source.sha256
case "$1" in
 prepare) "$py" prepare.py "$SLURM_ARRAY_TASK_ID";;
 smoke) "$py" prepare.py 2 smoke;;
 a6) "$py" a6.py "$SLURM_ARRAY_TASK_ID";;
 a6smoke) "$py" a6.py 2 smoke;;
 score) (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.txt
 grep -x OK replay.txt && "$py" score.py;;
esac
