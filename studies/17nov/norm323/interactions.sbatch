#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-323-interactions
#SBATCH --output=interactions-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s323_interactions_20261117
sha256sum -c interactions.sha256 >/dev/null
cd /mnt/lustre/work/gehler/gfh527/ni26_s323_normsplit_20261117
sha256sum -c source323.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I /mnt/lustre/work/gehler/gfh527/ni26_s323_interactions_20261117/interactions.py
