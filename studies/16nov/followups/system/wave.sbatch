#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-294
#SBATCH --output=logs/294-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes294_20261116/system
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c wave-src.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
arms=(E05CS E1CS E2CS D05CS D1CS D2CS E1CA D1CA E05 E2 D05 D2 FBL15 EFBL DFBL zero)
if [[ ${1:-} == smoke ]]; then
  for arm in zero E05CS FBL15; do "$PY" -B -I studies/16nov/followups/system/task.py "$arm" within0; done
else
  id=${SLURM_ARRAY_TASK_ID:?}
  "$PY" -B -I studies/16nov/followups/system/task.py "${arms[$((id / 8))]}" "${orders[$((id % 8))]}"
fi
