#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 5
#SBATCH -J ni26-f15e-held
#SBATCH -o held-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
origin=$base/ni26_s323_normsplit_20261117
f14=$base/ni26_friday_s1c_training_20261118
f15=$base/ni26_friday_s1c_skip_20261118
cross=$base/ni26_cross411b_20261118
PY=$base/neural-interfaces-2026/.venv/bin/python
cd "$base/ni26_friday_s1c_skip_external_20261118"
work=$PWD
sha256sum --quiet -c score.sha256
(cd "$origin" && sha256sum --quiet -c source323.sha256)
printf '%s\n' 'f9171d7228cce33215807cad7988438dd3e3879d4d3b1620e3cec4c17ea32386  score.sha256' | (cd "$f14" && sha256sum --quiet -c - && sha256sum --quiet -c score.sha256)
printf '%s\n' '9fbb2d8b80415c142219c776417a584d48c80451e94661b2d9201dd172394e2c  score.sha256' | (cd "$f15" && sha256sum --quiet -c - && sha256sum --quiet -c score.sha256)
(cd "$cross" && sha256sum --quiet -c "$work/orders.sha256")
export PYTHONPATH="$origin:$origin/experiments:$origin/studies/14nov/newmember:$f14:$f15:$cross"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
case "${1:?expected smoke, matrix or collect}" in
    smoke) "$PY" -B held_external.py 5 --smoke ;;
    matrix) "$PY" -B held_external.py "${SLURM_ARRAY_TASK_ID:?}" ;;
    collect) "$PY" -B collect_external.py ;;
    *) exit 2 ;;
esac
