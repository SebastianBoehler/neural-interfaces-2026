#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=ni26-laneP2-seg
#SBATCH --output=logs/seg-%A_%a.out
# usage (release dir): sbatch --array=0-9 laneP2/seg.sbatch PKG_ZIP   task i = held person P(i+1)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"; mkdir -p seg logs
held=$(printf "P%02d" $((SLURM_ARRAY_TASK_ID + 1)))
exec /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B laneP2/seg_member.py "$held" seg "${1:?}"
