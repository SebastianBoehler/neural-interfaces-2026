#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani,a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-l1-eeg
#SBATCH --output=logs/eeg-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/train
sha256sum -c inputs.sha256
sha256sum -c lane1-source.sha256
sha256sum -c eeg-batch.sha256
export NI26_WS="$PWD" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if (( SLURM_ARRAY_TASK_ID >= 6 )); then
    seed=$((SLURM_ARRAY_TASK_ID % 6))
    shopt -s nullglob
    files=(runs/diag/reve283/teacher_s${seed}_P*.npz)
    test "${#files[@]}" -eq 10
    manifest="logs/eeg-teacher-inputs-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sha256"
    test ! -e "$manifest"
    sha256sum "${files[@]}" > "$manifest"
fi
"$PY" -B -I studies/16nov/lane1/multistream.py "$SLURM_ARRAY_TASK_ID"
if (( SLURM_ARRAY_TASK_ID >= 6 )); then
    sha256sum -c "$manifest"
fi
