#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f32c-pupil
#SBATCH -o f32c-pupil-%A_%a.out
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
sha256sum --quiet -c pupil.sha256
(cd "$f32" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256 && sha256sum --quiet -c before_labels.sha256)
(cd "$c20" && sha256sum --quiet -c source.sha256)
ext="$base/ni26_friday_wave_motor_20261118/pretrain/physionet.sha256"
printf '%s\n' "b49b01b84ed61ad10865381a68c49d8e65b23ce99ed79527fc002071ad66a403  $ext" | sha256sum --quiet -c -
(cd "$base/neural-interfaces-2026/data/physionetmi" && sha256sum --quiet -c "$ext")
(cd "$origin" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
printf '%s\n' '503080a78e0c7b66f51e03451a93fd149613888dfa98328bce7c2795e63e018f  score.sha256' | \
    (cd "$f16" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256 && sha256sum --quiet -c before_labels.sha256)
printf '%s\n' '7ed98f96ba6b05165a1bc035a20af0d7dc516b9b1d61c82533abcc85ecbf2372  actor.sha256' | (cd "$f17" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
export PYTHONPATH="$PWD:$f32:$c20:$origin:$origin/experiments:$origin/studies/14nov/newmember:$f14:$f16:$f17"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
id=${SLURM_ARRAY_TASK_ID:?}
case "${1:?expected smoke or pupils}" in
    smoke) exec "$PY" -B produce32.py P01 "$id" smoke ;;
    pupils)
        for seed in 0 1; do [[ -s "smoke/P01_s$seed.npz" && ! -w "smoke/P01_s$seed.npz" ]]; done
        printf -v held 'P%02d' "$((id%10+1))"
        exec "$PY" -B produce32.py "$held" "$((id/10))" pupils ;;
    *) exit 2 ;;
esac
