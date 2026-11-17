#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s312-gap-audit
#SBATCH --output=audit-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_raw312_gap_audit_20261117
sha256sum -c audit.sha256 >/dev/null
cd /mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2
sha256sum -c source.sha256 >/dev/null
export NI26_WS="$PWD" NI26_ROOT="$PWD" NI26_CACHE=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_data1/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I /mnt/lustre/work/gehler/gfh527/ni26_raw312_gap_audit_20261117/report.py
