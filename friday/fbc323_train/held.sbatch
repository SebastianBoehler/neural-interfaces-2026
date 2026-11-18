#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 4
#SBATCH -J ni26-f14-held
#SBATCH -o held-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
origin=$base/ni26_s323_normsplit_20261117
PY=$base/neural-interfaces-2026/.venv/bin/python
cd "$base/ni26_friday_s1c_training_20261118"
sha256sum --quiet -c score.sha256
(cd "$origin" && sha256sum --quiet -c source323.sha256)
export PYTHONPATH="$origin:$origin/experiments:$origin/studies/14nov/newmember"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
case "${1:?expected smoke, matrix or collect}" in
    smoke) "$PY" -B held.py P01 0 --smoke ;;
    matrix)
        id=${SLURM_ARRAY_TASK_ID:?}
        printf -v person 'P%02d' "$((id%10+1))"
        "$PY" -B held.py "$person" "$((id/10))"
        ;;
    collect) "$PY" -B collect.py "${2:?expected blind inference array id}" ;;
    *) exit 2 ;;
esac
