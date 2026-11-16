#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=30-59%4
#SBATCH --job-name=ni26-l1-teachers
#SBATCH --output=logs/teacher-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/train
sha256sum -c inputs.sha256
sha256sum -c lane1-source.sha256
sha256sum -c reve-extra.sha256
export NI26_WS="$PWD" HF_HOME="$PWD/hf" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export REVE_POSITIONS_PATH="$PWD/mne_data"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
seed=$((SLURM_ARRAY_TASK_ID / 10))
printf -v held 'P%02d' "$((SLURM_ARRAY_TASK_ID % 10 + 1))"
test ! -e "runs/diag/reve283/teacher_s${seed}_${held}.npz"
"$PY" -B -I studies/15nov/grid283/teacher.py "$SLURM_ARRAY_TASK_ID"
