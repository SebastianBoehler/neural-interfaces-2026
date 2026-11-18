#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --job-name=ni26-f13-s1c
#SBATCH --output=f13-%A_%a.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
ACTOR=$BASE/ni26_friday_f13_20261118
ORIGIN=$BASE/ni26_s323_normsplit_20261117
PY=$BASE/neural-interfaces-2026/.venv/bin/python
cd "$ACTOR"
sha256sum --quiet -c source.sha256
(cd "$ORIGIN" && sha256sum --quiet -c source323.sha256)
export PYTHONPATH="$ORIGIN:$ORIGIN/experiments:$ORIGIN/studies/14nov/newmember"
export NI26_DEVICE=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ "$#" == 2 && "$1" == smoke ]]; then
    "$PY" -B replay_f13.py "$2" 0 --smoke
else
    id=${SLURM_ARRAY_TASK_ID:?}
    printf -v person 'P%02d' "$((id%10+1))"
    "$PY" -B replay_f13.py "$person" "$((id/10))"
fi
