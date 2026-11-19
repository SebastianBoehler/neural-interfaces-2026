#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-f13-score
#SBATCH --output=f13-score-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
ACTOR=$BASE/ni26_friday_f13_20261118
ORIGIN=$BASE/ni26_s323_normsplit_20261117
PY=$BASE/neural-interfaces-2026/.venv/bin/python
cd "$ACTOR"
sha256sum --quiet -c source.sha256
sha256sum --quiet -c score.sha256
(cd "$ORIGIN" && sha256sum --quiet -c source323.sha256)
export PYTHONPATH="$ORIGIN:$ORIGIN/experiments:$ORIGIN/studies/14nov/newmember"
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$PY" -B score_f13.py "${1:?expected blind producer array ID}"
