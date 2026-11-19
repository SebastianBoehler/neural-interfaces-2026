#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f37e
#SBATCH -o fit-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
f14=$(cd ../friday_s1c_training_20261118 && pwd -P)
f16=$(cd ../friday_s1c_distill_20261118 && pwd -P)
f16e=$(cd ../friday_s1c_distill_external_20261118 && pwd -P)
f17=$(cd ../friday_s1c_phase_20261118 && pwd -P)
f32e=$(cd ../friday_s1c_motor_external_20261119 && pwd -P)
sha256sum --quiet -c actor.sha256
(cd "$f14" && sha256sum --quiet -c actor.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256)
verify() (
    cd "$1"
    printf '%s  actor.sha256\n' "$2" | sha256sum --quiet -c -
    sha256sum --quiet -c actor.sha256
)
verify "$f16e" b4dd2f3eb7ae50c417334fed30d4a5f06553e28631892245e0e456983601441a
verify "$f17" 7ed98f96ba6b05165a1bc035a20af0d7dc516b9b1d61c82533abcc85ecbf2372
verify "$f32e" fdbc95d3649c0a55efb1c4de6fdcfcb19b8d278a6c319392a7e3872f1d90417e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$f14:$f14/original:$f14/original/experiments:$f16:$f16e:$f17:$f32e"
"$f14/runtime/bin/python" -m pip freeze --all | LC_ALL=C sort | cmp -s "$f14/runtime.lock" -
case "${1:?expected smoke or matrix}" in
    smoke) "$f14/runtime/bin/python" -B fit_all_source.py 0 --smoke ;;
    matrix) "$f14/runtime/bin/python" -B fit_all_source.py "${SLURM_ARRAY_TASK_ID:?}" ;;
    *) exit 2 ;;
esac
