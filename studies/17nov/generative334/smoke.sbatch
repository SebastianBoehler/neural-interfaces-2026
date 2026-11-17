#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s334-sourcesmoke
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export GEN_ROOT="$PWD" FIXED_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_gen334_v4_20261117/fixed_features NI26_DEVICE=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B smoke.py
