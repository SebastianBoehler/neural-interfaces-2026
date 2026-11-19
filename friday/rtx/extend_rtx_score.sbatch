#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 5
#SBATCH -J ni26-score-rtx-nine
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
work=$PWD
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$prior:$repo/experiments
sha256sum --quiet -c score.sha256
(cd "$prior" && sha256sum --quiet -c "$work/rtx_parents.sha256")
(cd "$base/ni26_friday_frozen_rtx_20261118" && sha256sum --quiet -c "$work/rtx_first_pair.sha256")
"$repo/.venv/bin/python" extend_rtx_score.py
