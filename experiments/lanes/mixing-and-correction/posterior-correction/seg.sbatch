#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --job-name=ni26-laneK2
#SBATCH --output=laneK2-%j.out
# usage (in $B5/laneK): sbatch seg.sbatch SEED
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
b=/mnt/lustre/work/gehler/gfh527
mkdir -p seg
exec $b/neural-interfaces-2026/.venv/bin/python -B seg_export.py "${1:-0}" seg
