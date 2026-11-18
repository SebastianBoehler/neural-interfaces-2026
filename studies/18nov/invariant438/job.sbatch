#!/bin/bash
#SBATCH --account=gehler
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-set438
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
case "${1:?}" in
 smoke) "$py" -B experiment.py P01 438 smoke ;;
 train) printf -v p 'P%02d' "$((SLURM_ARRAY_TASK_ID%10+1))"
        "$py" -B experiment.py "$p" "$((438+SLURM_ARRAY_TASK_ID/10))" ;;
 score) (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) > replay.log
        grep -Fx OK replay.log
        "$py" -B score.py ;;
 *) exit 2 ;;
esac
sha256sum -c source.sha256
