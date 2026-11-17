#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s337-optguard
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export COV_ROOT="$PWD" REFERENCE336_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_reference_bridge336_v1_20261117 NI26_DEVICE=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B optimization.py
