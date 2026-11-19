#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f28-teacher
#SBATCH -o teacher-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
f14=$(cd ../friday_s1c_training_20261118 && pwd -P)
f16=$(cd ../friday_s1c_distill_20261118 && pwd -P)
f17=$(cd ../friday_s1c_phase_20261118 && pwd -P)
f22=$(cd ../friday_s1c_temporal_v3_20261119 && pwd -P)
sha256sum --quiet -c teacher.sha256
printf '%s\n' '8a7d1784adbd88fd7381e9f3628a613fa1b7768c4732ca2551ae37cb9687f2c1  actor.sha256' | (cd "$f22" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
(cd "$f14" && sha256sum --quiet -c actor.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256)
printf '%s\n' '7ed98f96ba6b05165a1bc035a20af0d7dc516b9b1d61c82533abcc85ecbf2372  actor.sha256' | (cd "$f17" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
f25=$(cd ../friday_s1c_modulation_20261119 && pwd -P)
printf '%s\n' '589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c  actor.sha256' | (cd "$f25" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$f14:$f14/original:$f14/original/experiments:$f16:$f17:$f22:$f25"
"$f14/runtime/bin/python" -m pip freeze --all | LC_ALL=C sort | cmp -s "$f14/runtime.lock" -
case "${1:?expected smoke or matrix}" in
    smoke)
        for group in 0 1 2; do "$f14/runtime/bin/python" -B teach.py P01 0 "$group" --smoke; done
        ;;
    matrix)
        id=${SLURM_ARRAY_TASK_ID:?}
        printf -v held 'P%02d' "$(((id/3)%10+1))"
        "$f14/runtime/bin/python" -B teach.py "$held" "$((id/30))" "$((id%3))"
        ;;
    *) exit 2 ;;
esac
