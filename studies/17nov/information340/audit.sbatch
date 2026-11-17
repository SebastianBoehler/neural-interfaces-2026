#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s340-information
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export INFO_ROOT="$PWD" NI26_DEVICE=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TORCHINDUCTOR_CACHE_DIR="/tmp/ni26_inductor_${SLURM_JOB_ID}" TRITON_CACHE_DIR="/tmp/ni26_triton_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B audit.py
