#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 4
#SBATCH -J ni26-f14-s1c
#SBATCH -o fit-%A_%a.out
set -euo pipefail
cd ~/ni26/friday_s1c_training_20261118
sha256sum --quiet -c actor.sha256
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/original:$PWD/original/experiments"
runtime/bin/python -m pip freeze --all | LC_ALL=C sort | cmp -s runtime.lock -
if [[ "${1:?expected smoke or matrix}" == smoke ]]; then
    runtime/bin/python -B fit.py P01 0 --smoke
elif [[ "$1" == matrix ]]; then
    id=${SLURM_ARRAY_TASK_ID:?}
    printf -v held 'P%02d' "$((id%10+1))"
    runtime/bin/python -B fit.py "$held" "$((id/10))"
else
    exit 2
fi
