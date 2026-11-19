#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
root=$1
run=$2
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
export CS_OFFICIAL="$repo/2026-competition/tracks/bci_decoding"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$repo/.venv/bin/python" -B -u "$root/cleanslate/teamA/report.py" --cells "$run/cells.npz" \
    --cap250-new "$base/ni26_friday_c25d_held_v1_20261119/full0.npz" \
    --cap250-ret "$base/ni26_lane_a_returning_20261119/out" \
    --data "$repo/data/proteus_windows.npz" --out "$run/compare_cap250.json"
