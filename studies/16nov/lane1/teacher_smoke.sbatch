#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-l1-teacher-smoke
#SBATCH --output=logs/teacher-smoke-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/train
sha256sum -c inputs.sha256
sha256sum -c reve-extra.sha256
export NI26_WS="$PWD" HF_HOME="$PWD/hf" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export REVE_POSITIONS_PATH="$PWD/mne_data"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/15nov/grid283/teacher.py smoke
