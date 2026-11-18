#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani,v100-galvani
#SBATCH -c 1
#SBATCH --mem=5G
#SBATCH -t 6
#SBATCH -J ni26-step448s
set -eu
repo=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum -c collector.sha256 --status
(cd ../ni26_s318_coral_fbl_20261117 && "$repo/.venv/bin/python" replay/check.py) > replay.txt
grep -x OK replay.txt
PYTHONPATH="../ni26_qualify385_v1_20261118:$repo/experiments" "$repo/.venv/bin/python" collect.py
