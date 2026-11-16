#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-296-rolling
#SBATCH --output=logs/296-rolling-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes296_20261116/gated_v2
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c gated2-src.sha256
sha256sum -c gated2-score.sha256
sha256sum -c rolling-v2-src.sha256
sha256sum -c rolling-registration.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/gated2/rolling_v2.py "$1"
