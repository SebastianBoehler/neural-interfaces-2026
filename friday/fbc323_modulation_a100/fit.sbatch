#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -w galvani-cn238
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f25a-source
#SBATCH -o fit-%A_%a.out
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
o=$b/ni26_s323_normsplit_20261117
f14=$b/ni26_friday_s1c_training_20261118
f16=$b/ni26_friday_s1c_distill_20261118
f17=$b/ni26_friday_s1c_phase_20261118
f22=$b/ni26_friday_s1c_temporal_v3_20261119
f25=$b/ni26_friday_s1c_modulation_20261119
f33=$b/ni26_friday_s1c_covariance_a100_whole_20261119
canary=$b/ni26_friday_s1c_covariance_a100_20261119
labels=$b/ni26_friday_wave_band_20261118
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c actor.sha256
(cd "$o" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256)
(cd "$f17" && sha256sum --quiet -c actor.sha256)
verify() (
    cd "$1"
    printf '%s  %s\n' "$2" "$3" | sha256sum --quiet -c -
    sha256sum --quiet -c "$3"
)
verify "$f22" 8a7d1784adbd88fd7381e9f3628a613fa1b7768c4732ca2551ae37cb9687f2c1 actor.sha256
verify "$f25" 589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c actor.sha256
verify "$f33" 94598f4eb485b20ea40c0d7c13f4497ed07d04e9e9c062c72953e97460b782bc actor.sha256
verify "$canary" 84511237df30a0ffc25c1abe096b9c9b539418b07bbf8acab2dfa821f54186dc canary.sha256
export PYTHONPATH="$f25:$o:$o/experiments:$f14:$f16:$f17:$f22:$f33:$labels"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python=$b/neural-interfaces-2026/.venv/bin/python
case "${1:?expected smoke or matrix}" in
    smoke) "$python" -B fit_f25a.py P01 0 --smoke ;;
    matrix)
        i=${SLURM_ARRAY_TASK_ID:?}
        printf -v held 'P%02d' "$((i%10+1))"
        "$python" -B fit_f25a.py "$held" "$((i/10))"
        ;;
    *) exit 2 ;;
esac
