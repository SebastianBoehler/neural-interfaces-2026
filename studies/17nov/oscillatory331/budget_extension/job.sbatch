#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:15:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s331-source-full
set -euo pipefail
HERE=/mnt/lustre/work/gehler/gfh527/ni26_oscillatory331_budget_20261117
cd "$HERE"
sha256sum -c budget.sha256 >/dev/null
export OSC_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_oscillatory331_v2_20261117
export NI26_DEVICE=cpu OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
/usr/bin/time -f 'COMPLETE_SOURCE_SECONDS=%e MAXRSS_KB=%M' srun --cpu-bind=cores "$PY" -B "$HERE/run.py"
