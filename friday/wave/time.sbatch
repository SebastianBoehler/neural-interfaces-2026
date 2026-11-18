#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 5
#SBATCH -J ni26-wave-time
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum --quiet -c actor.sha256
if [[ $1 == smoke ]]; then
    shift
    for kind in "$@"; do
        ../venv/bin/python fit.py "$kind" 460 P01 --smoke
    done
else
    seeds=(460 1460)
    task=${SLURM_ARRAY_TASK_ID:?}
    seed=${seeds[task/10]}
    printf -v held 'P%02d' "$(( task % 10 + 1 ))"
    ../venv/bin/python fit.py "$1" "$seed" "$held"
fi
