#!/bin/bash
#SBATCH --job-name=ni26-js321-returning-sinc
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
index=${SLURM_ARRAY_TASK_ID:?}
if ((index<5)); then fold=P04_s23; seed=$((index+1)); else fold=P10_s23; seed=$((index-5)); fi
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python returning_train.py "$fold" "$seed"
