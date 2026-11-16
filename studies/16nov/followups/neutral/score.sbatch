#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-enu-score
#SBATCH --output=logs/neutral-score-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s302_neutral_20261116/neutral
for f in chain-src.sha256 stage-src.sha256 followup-src.sha256 repair-src.sha256 repair-v3-src.sha256 neutral-src.sha256 neutral-registration.sha256 neutral-reuse.sha256 neutral-final.sha256; do sha256sum -c "$f"; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/neutral/score.py
