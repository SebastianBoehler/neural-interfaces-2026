#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,cpu-galvani
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 5
#SBATCH -J ni26-score-c8
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
work=$PWD
export PYTHONPATH=$prior:$repo/experiments
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c --status source.sha256
(cd "$prior" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status inputs.sha256)
grep -Fx OK replay.log
"$repo/.venv/bin/python" collect.py
