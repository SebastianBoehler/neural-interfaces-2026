#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-laneT-retcollect
#SBATCH --output=logs/retcollect-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B ttt_collect.py out summary.json
