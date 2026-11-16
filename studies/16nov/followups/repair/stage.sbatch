#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=ni26-302-stages
#SBATCH --output=logs/302-stages-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes302_20261116/repair
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c repair-src.sha256
sha256sum -c repair-v3-src.sha256
sha256sum -c repair-score.sha256
sha256sum -c repair-stage.sha256
sha256sum -c repair-registration.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/repair/stage_job.py
