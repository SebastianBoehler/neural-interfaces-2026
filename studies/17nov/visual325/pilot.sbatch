#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ni26-s325-source
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export VISUAL_ROOT="$PWD" NI26_DEVICE=cuda OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export XFORMERS_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
srun --cpu-bind=cores "$PY" -B pilot325.py
