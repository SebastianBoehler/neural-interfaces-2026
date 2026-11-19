#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 6
#SBATCH -J ni26-fc11c
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
work=$PWD
export PYTHONPATH=$prior:$repo/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c --status source.sha256
(cd "$prior" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status inputs.sha256)
sha256sum -c --status smoke.sha256
"$repo/.venv/bin/python" collect.py
