#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s331-source
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OSC_ROOT="$PWD" NI26_DEVICE=cpu OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B pilot331.py
