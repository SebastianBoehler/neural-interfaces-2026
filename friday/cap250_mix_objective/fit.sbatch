#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -w galvani-cn238
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 8
#SBATCH -J ni26-c31d-fit
#SBATCH -o fit-%A_%a.out
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
o=$b/ni26_s323_normsplit_20261117
f14=$b/ni26_friday_s1c_training_20261118
f16=$b/ni26_friday_s1c_distill_20261118
f35=$b/ni26_friday_wave_band_20261118
tape=$b/ni26_friday_c31c_source_mix_v1_20261119
hard=$b/ni26_friday_s1c_motor_external_cpu_20261119
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c fit.sha256
(cd "$o" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256)
(cd "$tape" && sha256sum --quiet -c source.sha256 && sha256sum --quiet -c seal.sha256)
(cd "$hard" && printf '%s\n' 'fdbc95d3649c0a55efb1c4de6fdcfcb19b8d278a6c319392a7e3872f1d90417e  actor.sha256' | sha256sum --quiet -c -
 sha256sum --quiet -c actor.sha256)
export PYTHONPATH="$o:$o/experiments:$f14:$f16:$f35"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mode=${1:?}
case "$mode" in
    smoke) arm=smoke ;;
    fit)
        test -s fit-smoke.json
        names=(hard nominal matched)
        arm=${names[${SLURM_ARRAY_TASK_ID:?}]}
        ;;
    *) exit 2;;
esac
exec "$b/neural-interfaces-2026/.venv/bin/python" -B train_mix.py "$arm"
