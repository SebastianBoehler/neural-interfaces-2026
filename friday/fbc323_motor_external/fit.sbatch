#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f32e
#SBATCH -o fit-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
f14=$(cd ../friday_s1c_training_20261118 && pwd -P)
f16=$(cd ../friday_s1c_distill_20261118 && pwd -P)
f16e=$(cd ../friday_s1c_distill_external_20261118 && pwd -P)
f32=$(cd ../friday_s1c_motor_joint_20261119 && pwd -P)
sha256sum --quiet -c actor.sha256
(cd "$f14" && sha256sum --quiet -c actor.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256)
printf '%s\n' 'b4dd2f3eb7ae50c417334fed30d4a5f06553e28631892245e0e456983601441a  actor.sha256' | (cd "$f16e" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
printf '%s\n' 'f8d074e6867ebfda82383c164c257d64f154b251b0ccdbcb5153b297e7917723  actor.sha256' | (cd "$f32" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$f14:$f14/original:$f14/original/experiments:$f16:$f16e:$f32"
"$f14/runtime/bin/python" -m pip freeze --all | LC_ALL=C sort | cmp -s "$f14/runtime.lock" -
case "${1:?expected smoke or matrix}" in
    smoke) "$f14/runtime/bin/python" -B fit_external.py 0 --smoke ;;
    matrix) "$f14/runtime/bin/python" -B fit_external.py "${SLURM_ARRAY_TASK_ID:?}" ;;
    *) exit 2 ;;
esac
