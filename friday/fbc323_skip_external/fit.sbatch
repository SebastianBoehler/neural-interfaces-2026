#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 4
#SBATCH -J ni26-f15e
#SBATCH -o fit-%A_%a.out
set -euo pipefail
cd ~/ni26/friday_s1c_skip_external_20261118
parent=$(cd ../friday_s1c_training_20261118 && pwd -P)
f15=$(cd ../friday_s1c_skip_20261118 && pwd -P)
sha256sum --quiet -c actor.sha256
(cd "$parent" && sha256sum --quiet -c actor.sha256)
(cd "$f15" && sha256sum --quiet -c actor.sha256)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$parent:$parent/original:$parent/original/experiments:$f15"
"$parent/runtime/bin/python" -m pip freeze --all | LC_ALL=C sort | cmp -s "$parent/runtime.lock" -
case "${1:?expected smoke or matrix}" in
    smoke) "$parent/runtime/bin/python" -B produce_external.py 0 --smoke ;;
    matrix) "$parent/runtime/bin/python" -B produce_external.py "${SLURM_ARRAY_TASK_ID:?}" ;;
    *) exit 2 ;;
esac
