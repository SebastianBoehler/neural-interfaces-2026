#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-300-collect
#SBATCH --output=logs/300-sharded-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s300_shards_20261116/system
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c followup-score.sha256
sha256sum -c attribution-src.sha256
sha256sum -c shard-src.sha256
sha256sum -c collect-v2-src.sha256
sha256sum -c attribution-registration.sha256
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/attribution/collect.py
