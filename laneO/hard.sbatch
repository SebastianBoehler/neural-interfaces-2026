#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 40
#SBATCH -J ni26-laneO-hard
#SBATCH -o hard-%A_%a.out
# usage (cd release dir with hard_orders.py): sbatch --array=0-19 hard.sbatch [smoke]    id -> held P(id%10+1), seed id/10
# environment copied from friday/f32c_v2/pupil.sbatch (pins verified, CPU only)
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
origin=$base/ni26_s323_normsplit_20261117
f14=$base/ni26_friday_s1c_training_20261118
f16=$base/ni26_friday_s1c_distill_20261118
f17=$base/ni26_friday_s1c_phase_20261118
f32=$base/ni26_friday_s1c_motor_joint_20261119
c20=$base/ni26_friday_causal_rebuild_20261118/fc20_pupils_v2
PY=$base/neural-interfaces-2026/.venv/bin/python
cd "${SLURM_SUBMIT_DIR:?}"
(cd "$f32" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c before_labels.sha256)
export PYTHONPATH="$PWD:$base/ni26_friday_causal_rebuild_20261118/f32c_v2:$f32:$c20:$origin:$origin/experiments:$origin/studies/14nov/newmember:$f14:$f16:$f17"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
id=${SLURM_ARRAY_TASK_ID:?}
printf -v held 'P%02d' "$((id%10+1))"
exec "$PY" -B hard_orders.py "$held" "$((id/10))" ${1:-}
