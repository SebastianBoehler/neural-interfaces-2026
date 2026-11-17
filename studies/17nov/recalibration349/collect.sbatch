#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s349-guarded-collect
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export RECAL349_ROOT="$PWD" NI26_DEVICE=cpu CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B collect349.py
