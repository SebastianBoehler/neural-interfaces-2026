#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 4
#SBATCH -J ni26-f15-s1c
#SBATCH -o fit-%A_%a.out
set -euo pipefail
cd ~/ni26/friday_s1c_skip_20261118
parent=$(cd ../friday_s1c_training_20261118 && pwd -P)
sha256sum --quiet -c actor.sha256
(cd "$parent" && sha256sum --quiet -c actor.sha256)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$parent:$parent/original:$parent/original/experiments"
"$parent/runtime/bin/python" -m pip freeze --all | LC_ALL=C sort | cmp -s "$parent/runtime.lock" -
if [[ "${1:?expected smoke or matrix}" == smoke ]]; then
    "$parent/runtime/bin/python" -B produce.py P01 0 --smoke
elif [[ "$1" == matrix ]]; then
    id=${SLURM_ARRAY_TASK_ID:?}
    printf -v held 'P%02d' "$((id%10+1))"
    "$parent/runtime/bin/python" -B produce.py "$held" "$((id/10))"
else
    exit 2
fi
