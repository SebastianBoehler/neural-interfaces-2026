#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -w galvani-cn238
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f33a-source
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
labels=$b/ni26_friday_wave_band_20261118
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
export PYTHONPATH="$o:$o/experiments:$f14:$f16:$f17:$f33:$canary:$labels"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "${1:?expected smoke or matrix}" in
    smoke) "$r/.venv/bin/python" -B fit_a100.py P01 0 --smoke ;;
    matrix)
        i=${SLURM_ARRAY_TASK_ID:?}
        printf -v held 'P%02d' "$((i%10+1))"
        "$r/.venv/bin/python" -B fit_a100.py "$held" "$((i/10))"
        ;;
    *) exit 2 ;;
esac
