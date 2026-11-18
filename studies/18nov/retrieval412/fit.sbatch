#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-retr412
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=../venv/bin/python
if [[ "$1" == smoke ]]; then
    for arm in real destroyed; do "$py" -B train.py "$arm" 412 smoke; done
else
    arms=(real destroyed real destroyed)
    seeds=(412 412 1412 1412)
    id=$SLURM_ARRAY_TASK_ID
    "$py" -B train.py "${arms[$id]}" "${seeds[$id]}"
fi
