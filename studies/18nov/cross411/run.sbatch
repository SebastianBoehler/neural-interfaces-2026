#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani,v100-galvani
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 3
#SBATCH -J ni26-cross411
set -eu
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
bank=$base/ni26_qualify385_v1_20261118
export OPENBLAS_NUM_THREADS=1 PYTHONPATH="$bank"
if [[ ${1:-} == --smoke ]]; then
    "$repo/.venv/bin/python" score.py "$bank" "$repo/data/scherer2015/windows.npz" --smoke
else
    (cd "$base/ni26_s318_coral_fbl_20261117" && "$repo/.venv/bin/python" replay/check.py) > replay.txt
    grep -x OK replay.txt
    "$repo/.venv/bin/python" score.py "$bank" "$repo/data/scherer2015/windows.npz" > result.json
fi
