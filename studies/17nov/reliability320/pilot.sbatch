#!/bin/bash
#SBATCH --job-name=ni26-reliability320-source
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=none
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export NI26_ROOT="$PWD" NI26_DEVICE=cpu OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=cores .venv/bin/python pilot.py
