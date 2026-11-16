#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-302-neutral
#SBATCH --output=logs/neutral-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s302_neutral_20261116/neutral
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c repair-src.sha256
sha256sum -c repair-v3-src.sha256
sha256sum -c neutral-src.sha256
sha256sum -c neutral-registration.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
if [[ ${1:-} == smoke ]]; then
 for arm in ZR ENU; do "$PY" -B -I studies/16nov/followups/neutral/task.py "$arm" within0; done
else
 "$PY" -B -I studies/16nov/followups/neutral/task.py ENU "${orders[${SLURM_ARRAY_TASK_ID:?}]}"
fi
