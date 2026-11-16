#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-302
#SBATCH --output=logs/302-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes302_20261116/repair
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c repair-src.sha256
sha256sum -c repair-v3-src.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
arms=(ET05 ET2 EPR25 EPR50 ZR)
if [[ ${1:-} == smoke ]]; then
  for arm in ZR TI PI ET2 EPR25; do "$PY" -B -I studies/16nov/followups/repair/task_v3.py "$arm" within0; done
else
  id=${SLURM_ARRAY_TASK_ID:?}
  "$PY" -B -I studies/16nov/followups/repair/task_v3.py "${arms[$((id / 8))]}" "${orders[$((id % 8))]}"
fi
