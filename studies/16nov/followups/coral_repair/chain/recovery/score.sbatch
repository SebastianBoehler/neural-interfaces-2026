#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-304-recovery-score
#SBATCH --output=logs/304-recovery-score-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s304_coral_chain_recovery_20261117
for f in chain304-src.sha256 stage-src.sha256 coral304-chain-src.sha256 recovery.sha256 runs/coral304_inputs/registration.sha256; do sha256sum -c "$f" >/dev/null; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/coral_repair/chain/recovery/score_guard.py
