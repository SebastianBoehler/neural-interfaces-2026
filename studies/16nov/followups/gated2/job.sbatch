#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-296
#SBATCH --output=logs/296-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes296_20261116/gated_v2
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c gated2-src.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
arms=(EG06 EG08 DG06 DG08)
if [[ ${1:-} == smoke ]]; then
  for arm in ZG06 EG06 DG08; do "$PY" -B -I studies/16nov/followups/gated2/task.py "$arm" within0; done
else
  id=${SLURM_ARRAY_TASK_ID:?}
  "$PY" -B -I studies/16nov/followups/gated2/task.py "${arms[$((id / 8))]}" "${orders[$((id % 8))]}"
fi
