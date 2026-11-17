#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-323-producer
#SBATCH --output=produce-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s323_normsplit_20261117
sha256sum -c source323.sha256 >/dev/null
export PYTHONPATH="$PWD:$PWD/experiments:$PWD/studies/14nov/newmember" NI26_DEVICE=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
id=${SLURM_ARRAY_TASK_ID:?}; printf -v person 'P%02d' "$((id%10+1))"; seed=$((id/10))
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B produce.py "$person" "$seed"
"$PY" -B verify_fold.py "$person" "$seed"
