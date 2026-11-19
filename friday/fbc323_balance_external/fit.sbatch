#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -w galvani-cn238
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f35e-source
#SBATCH -o fit-%A_%a.out
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
r=$b/neural-interfaces-2026
o=$b/ni26_s323_normsplit_20261117
f14=$b/ni26_friday_s1c_training_20261118
f16=$b/ni26_friday_s1c_distill_20261118
f17=$b/ni26_friday_s1c_phase_20261118
f33=$b/ni26_friday_s1c_covariance_v2_20261119
canary=$b/ni26_friday_s1c_covariance_a100_20261119
f33a=$b/ni26_friday_s1c_covariance_a100_whole_20261119
labels=$b/ni26_friday_wave_band_20261118
f35=$b/ni26_friday_s1c_balance_a100_20261119
f25e=$b/ni26_friday_s1c_modulation_external_a100_20261119
f32e=$b/ni26_friday_s1c_motor_external_cpu_20261119
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c actor.sha256
(cd "$o" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256)
(cd "$f17" && sha256sum --quiet -c actor.sha256)
printf '%s\n' '984e6a7c7776fefca6fd70924ddc135b92dd53ab3fdffd206fc435ccaa5df70f  actor.sha256' | \
    (cd "$f33" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
printf '%s\n' '84511237df30a0ffc25c1abe096b9c9b539418b07bbf8acab2dfa821f54186dc  canary.sha256' | \
    (cd "$canary" && sha256sum --quiet -c - && sha256sum --quiet -c canary.sha256)
printf '%s\n' '94598f4eb485b20ea40c0d7c13f4497ed07d04e9e9c062c72953e97460b782bc  actor.sha256' \
               '6c8e27573d2b91176ec6d2e98068e94d8e671d878c92e515f6c58899ce558dbe  score.sha256' | \
    (cd "$f33a" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
printf '%s\n' 'b153df05e923c0b40795adaed413d8d024087e49e4b2be51445d81b6180db42e  actor.sha256' | \
    (cd "$f35" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
(cd "$f32e" && sha256sum --quiet -c actor.sha256)
(cd "$f25e" && sha256sum --quiet -c artifact.sha256)
export PYTHONPATH="$o:$o/experiments:$f14:$f16:$f17:$f33:$f33a:$canary:$labels:$f35:$f32e:$f32e/anchors_f16e"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "${1:?expected smoke or matrix}" in
    smoke) "$r/.venv/bin/python" -B fit_external.py 0 --smoke ;;
    matrix) "$r/.venv/bin/python" -B fit_external.py "${SLURM_ARRAY_TASK_ID:?}" ;;
    *) exit 2 ;;
esac
