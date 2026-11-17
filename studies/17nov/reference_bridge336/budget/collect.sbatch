#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s336-complete
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_reference_bridge336_budget_20261117
sha256sum -c budget.sha256 >/dev/null
export BRIDGE_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_reference_bridge336_v1_20261117
export NI26_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B run.py collect
