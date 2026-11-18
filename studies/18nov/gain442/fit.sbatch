#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-gain442
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=../venv/bin/python
if [[ "$1" == smoke ]]; then
    for arm in clean coherent independent; do "$py" -B train.py "$arm" 382 smoke; done
else
    arms=(clean coherent independent clean coherent independent)
    seeds=(382 382 382 1382 1382 1382)
    "$py" -B train.py "${arms[$SLURM_ARRAY_TASK_ID]}" "${seeds[$SLURM_ARRAY_TASK_ID]}"
fi
