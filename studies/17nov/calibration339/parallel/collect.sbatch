#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-s339-collect
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_calibration339_parallel_v1_20261117
sha256sum -c source.sha256 >/dev/null
export CAL_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_calibration339_parallel_v1_20261117
export NI26_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="${CAL_ROOT}/runtime-cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B collect339.py
