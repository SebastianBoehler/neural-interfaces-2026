#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 8
#SBATCH -J protect436score
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
py=$repo/.venv/bin/python
sha256sum --quiet -c source.sha256
sha256sum --quiet -c collector.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
(cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay_check.txt
grep -x OK replay_check.txt
"$py" collect.py "$1"
