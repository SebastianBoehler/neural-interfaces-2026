#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --gres=none
#SBATCH --array=0-1%2
#SBATCH --job-name=ni26-s347-supervised-crops
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export TRAIN347_ROOT="$PWD" NI26_DEVICE=cpu CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B produce347.py "${SLURM_ARRAY_TASK_ID:?}"
