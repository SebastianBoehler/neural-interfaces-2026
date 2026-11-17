#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-s339-produce
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_calibration339_parallel_v1_20261117
sha256sum -c source.sha256 >/dev/null
export CAL_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_calibration339_parallel_v1_20261117
export NI26_DEVICE=cuda OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="${CAL_ROOT}/runtime-cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B produce339.py "${SLURM_ARRAY_TASK_ID:?}"
