#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani,a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --job-name=ni26-l1-same-model
#SBATCH --output=logs/same-model-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/train
sha256sum -c inputs.sha256
sha256sum -c lane1-source.sha256
sha256sum -c same-model-source.sha256
export NI26_WS="$PWD" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/lane1/eegn_same_model.py "$SLURM_ARRAY_TASK_ID"
