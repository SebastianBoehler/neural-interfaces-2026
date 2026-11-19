#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -w galvani-cn211
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f25e-source
#SBATCH -o fit-%A_%a.out
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
repo=$b/neural-interfaces-2026
o=$b/ni26_s323_normsplit_20261117
f14=$b/ni26_friday_s1c_training_20261118
f16=$b/ni26_friday_s1c_distill_20261118
f22=$b/ni26_friday_s1c_temporal_v3_20261119
f25=$b/ni26_friday_s1c_modulation_20261119
f32e=$b/ni26_friday_s1c_motor_external_cpu_20261119
labels=$b/ni26_friday_wave_band_20261118
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c actor.sha256
(cd "$o" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
printf '%s\n' '8a7d1784adbd88fd7381e9f3628a613fa1b7768c4732ca2551ae37cb9687f2c1  actor.sha256' | \
    (cd "$f22" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
printf '%s\n' '589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c  actor.sha256' | \
    (cd "$f25" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
(cd "$f32e" && sha256sum --quiet -c actor.sha256)
export PYTHONPATH="$o:$o/experiments:$f14:$f16:$f25:$f22:$f32e:$f32e/anchors_f16e:$labels"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "${1:?expected smoke or matrix}" in
    smoke) "$repo/.venv/bin/python" -B produce_external.py 0 --smoke ;;
    matrix) "$repo/.venv/bin/python" -B produce_external.py "${SLURM_ARRAY_TASK_ID:?}" ;;
    *) exit 2 ;;
esac
