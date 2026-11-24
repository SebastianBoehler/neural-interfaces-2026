#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 20
#SBATCH -J ni26-laneO-score
#SBATCH -o score-%j.out
# usage (release dir ni26_laneO_20261120): sbatch [--dependency=afterok:..] score_o.sbatch
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
cd "${SLURM_SUBMIT_DIR:?}"
ls hard/hard_P??_s?.npz | wc -l | grep -qx 20
exec $b/neural-interfaces-2026/.venv/bin/python -B score_o.py banks hard score_o.json score_o.md
