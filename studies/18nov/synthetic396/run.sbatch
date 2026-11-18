#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH -c 2
#SBATCH --mem=5G
#SBATCH -t 00:06:00
#SBATCH -J ni26-synth396
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
if [[ "$1" == smoke ]]; then
  (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) > replay.log
  grep -Fx OK replay.log
fi
"$py" study.py "$1" "${SLURM_ARRAY_TASK_ID:-0}"
