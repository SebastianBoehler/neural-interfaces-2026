#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s338-guard
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_combinations338_v2_20261117
sha256sum -c source.sha256 >/dev/null
export COMBO_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_combinations338_v2_20261117
export NI26_DEVICE=cpu OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export TORCHINDUCTOR_CACHE_DIR="${COMBO_ROOT}/runtime-cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B guard338.py
