#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-galvani,2080-preemptable-galvani,a100-galvani,a100-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:06:00
#SBATCH --job-name=ni26-fuse383
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
case "$1" in
smoke)
  (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) > replay.log
  grep -Fx OK replay.log
  "$py" -B experiment.py smoke ;;
seed)
  seeds=(382 1382)
  grep -Fx OK replay.log
  "$py" -B experiment.py "${seeds[${SLURM_ARRAY_TASK_ID:?}]}" ;;
collect)
  "$py" -B collect.py ;;
esac
sha256sum -c source.sha256
