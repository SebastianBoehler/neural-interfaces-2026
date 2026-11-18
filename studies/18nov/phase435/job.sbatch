#!/bin/bash
#SBATCH --account=gehler
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-phase435
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
sha256sum -c source.sha256
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
case ${1:?} in
 smoke) "$py" -B experiment.py P01 smoke ;;
 train) printf -v p 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"; "$py" -B experiment.py "$p" ;;
 score) (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) > replay.log
        grep -Fx OK replay.log
        "$py" -B score.py ;;
 *) exit 2 ;;
esac
sha256sum -c source.sha256
