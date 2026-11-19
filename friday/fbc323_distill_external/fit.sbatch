#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 4
#SBATCH -J ni26-f16e
#SBATCH -o fit-%A_%a.out
set -euo pipefail
cd ~/ni26/friday_s1c_distill_external_20261118
f14=$(cd ../friday_s1c_training_20261118 && pwd -P)
f15e=$(cd ../friday_s1c_skip_external_20261118 && pwd -P)
f16=$(cd ../friday_s1c_distill_20261118 && pwd -P)
sha256sum --quiet -c actor.sha256
for parent in "$f14" "$f15e" "$f16"; do
    (cd "$parent" && sha256sum --quiet -c actor.sha256)
done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$f14:$f14/original:$f14/original/experiments:$f16"
"$f14/runtime/bin/python" -m pip freeze --all | LC_ALL=C sort | cmp -s "$f14/runtime.lock" -
case "${1:?expected smoke or matrix}" in
    smoke) "$f14/runtime/bin/python" -B produce_external_kd.py 0 --smoke ;;
    matrix) "$f14/runtime/bin/python" -B produce_external_kd.py "${SLURM_ARRAY_TASK_ID:?}" ;;
    *) exit 2 ;;
esac
