#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-f25c-pupil
#SBATCH --output=f25c-pupil-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
origin=$base/ni26_s323_normsplit_20261117
f14=$base/ni26_friday_s1c_training_20261118
f16=$base/ni26_friday_s1c_distill_20261118
f17=$base/ni26_friday_s1c_phase_20261118
f25=$base/ni26_friday_s1c_modulation_20261119
camp=$base/ni26_friday_causal_rebuild_20261118
c20=$camp/fc20_pupils_v2
repo=$base/neural-interfaces-2026
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
(cd "$origin" && sha256sum --check --status source323.sha256)
(cd "$f14" && sha256sum --check --status score.sha256)
(cd "$f16" && sha256sum --check --status actor.sha256 && sha256sum --check --status score.sha256)
(cd "$f17" && sha256sum --check --status actor.sha256 && sha256sum --check --status score.sha256 && sha256sum --check --status before_labels.sha256)
printf '%s\n' '589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c  actor.sha256' | (cd "$f25" && sha256sum --check --status - && sha256sum --check --status actor.sha256 && sha256sum --check --status score.sha256 && sha256sum --check --status before_labels.sha256)
printf '%s\n' '5925163a6331fc062a51f80f8262c05f7c12f069cfdba86d9e2cddf1f57d349c  source.sha256' | (cd "$c20" && sha256sum --check --status - && sha256sum --check --status source.sha256)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$work:$f25:$f17:$c20:$origin:$origin/experiments:$origin/studies/14nov/newmember:$f14:$f16"
task=${SLURM_ARRAY_TASK_ID:?}
if [[ ${1:?} == smoke ]]; then
    exec "$repo/.venv/bin/python" -B produce_pupils.py P01 "$task" smoke
fi
[[ $1 == pupils ]]
printf -v person 'P%02d' "$((task%10+1))"
exec "$repo/.venv/bin/python" -B produce_pupils.py "$person" "$((task/10))" pupils
