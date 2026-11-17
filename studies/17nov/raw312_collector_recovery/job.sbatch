#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s312-collect-recover
#SBATCH --output=collector-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_raw312_collector_recovery_20261117
sha256sum -c recovery.sha256 >/dev/null
cd /mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2
sha256sum -c source.sha256 >/dev/null
test ! -e whole_results.json
export NI26_WS="$PWD" NI26_ROOT="$PWD" NI26_CACHE=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_data1/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I /mnt/lustre/work/gehler/gfh527/ni26_raw312_collector_recovery_20261117/run.py
