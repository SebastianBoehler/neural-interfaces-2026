#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s324-guard
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export VISUAL_ROOT="$PWD" OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export XFORMERS_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
srun --cpu-bind=cores "$PY" -B guard.py
