#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 6
#SBATCH -J ni26-error-teacher
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
export PYTHONPATH="$base/ni26_match456_v2_20261118:$base/ni26_history443_20261118:$base/ni26_wave379_v1_20261118:$base/ni26_fusion383_v1_20261118"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c --status export.sha256
(cd "$base/ni26_s318_coral_fbl_20261117" && "$repo/.venv/bin/python" replay/check.py) > teacher_replay.log
grep -Fx OK teacher_replay.log
"$repo/.venv/bin/python" export_teacher.py
