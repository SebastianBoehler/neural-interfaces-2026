#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-kernel405
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
case "$1" in
 smoke) "$py" -B fit.py P01 405 smoke ;;
 fit) printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"; "$py" -B fit.py "$person" "$2" ;;
 score) (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" -B replay/check.py) > "replay-$2.log"
        grep -Fx OK "replay-$2.log"
        "$py" -B score.py "$2" ;;
 *) exit 2 ;;
esac
