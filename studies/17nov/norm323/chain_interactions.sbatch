#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-323-chain-interactions
#SBATCH --output=chain-interactions-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s323_interactions_20261117
sha256sum -c chain-interactions.sha256 >/dev/null
cd /mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117
sha256sum -c representation-source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I /mnt/lustre/work/gehler/gfh527/ni26_s323_interactions_20261117/chain_interactions.py
