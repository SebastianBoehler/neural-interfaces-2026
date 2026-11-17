#!/bin/bash
#SBATCH --job-name=ni26-js321-shared-guard
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_shared2
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python guard_shared.py
