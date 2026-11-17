#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-scale380
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
sha256sum --quiet -c source.sha256
root=/mnt/lustre/work/gehler/gfh527
py="$root/neural-interfaces-2026/.venv/bin/python"
case ${1:?} in
 smoke) "$py" -B pilot.py P01 smoke ;;
 train) printf -v p 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"; "$py" -B pilot.py "$p" ;;
 score) (cd "$root/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
        grep -Fx OK replay.log; "$py" -B score.py ;;
 *) exit 2 ;;
esac
sha256sum --quiet -c source.sha256
