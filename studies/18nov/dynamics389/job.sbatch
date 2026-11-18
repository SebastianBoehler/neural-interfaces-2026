#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-galvani,2080-preemptable-galvani,a100-galvani,a100-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-dyn389
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
case "$1" in
 inputsmoke) "$py" -B features.py smoke ;;
 input) "$py" -B features.py full ;;
 smoke) "$py" -B fold.py P01 "$2" smoke ;;
 fit) person=$(printf 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"); "$py" -B fold.py "$person" "$2" ;;
 score)
   (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) > replay.log
   grep -Fx OK replay.log
   "$py" -B score389.py ;;
esac
