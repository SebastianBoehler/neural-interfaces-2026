#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-296-zero
#SBATCH --output=logs/296-zero-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes296_20261116/gated_v2
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c gated2-src.sha256
sha256sum -c zero-src.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/gated2/task.py ZG06 "${orders[${SLURM_ARRAY_TASK_ID:?}]}"
