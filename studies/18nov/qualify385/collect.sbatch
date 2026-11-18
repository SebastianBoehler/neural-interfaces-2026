#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani,v100-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 8
#SBATCH -J ni26-qual385-score
set -eu
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
sha256sum -c sources.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
(cd "$base/ni26_s318_coral_fbl_20261117" && "$repo/.venv/bin/python" replay/check.py) > replay-proof.txt
grep -x OK replay-proof.txt
PYTHONPATH="$repo/experiments" "$repo/.venv/bin/python" collect.py
