#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s347-source-controls
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export TRAIN347_ROOT="$PWD" NI26_DEVICE=cpu CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B smoke347.py
