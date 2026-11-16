#!/bin/bash
#SBATCH --job-name=ni26-s312-pilot
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --gres=none
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256
export NI26_ROOT=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
export NI26_DEVICE=cpu
export NI26_OUT="$PWD/output" OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 HF_HUB_OFFLINE=1
srun --cpu-bind=cores "$NI26_ROOT/.venv/bin/python" pilot.py
