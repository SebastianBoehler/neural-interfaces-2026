#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=7G
#SBATCH -t 5
#SBATCH -J ni26-error-fit
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum --quiet -c actor.sha256
objectives=(true wrongswap ordinary)
seeds=(460 1460)
if [[ $1 == smoke ]]; then
    for objective in "${objectives[@]}"; do
        ../venv/bin/python error_fit.py "$objective" 460 --smoke
    done
else
    task=${SLURM_ARRAY_TASK_ID:?}
    ../venv/bin/python error_fit.py "${objectives[task % 3]}" "${seeds[task / 3]}"
fi
