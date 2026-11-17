#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s312-allcpu-score
set -euo pipefail
HERE=/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_finish_20261117
cd "$HERE"
sha256sum -c finish-source.sha256 >/dev/null
ROOT=/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2
cd "$ROOT"
sha256sum -c source.sha256 >/dev/null
test ! -e whole_results.json
export NI26_WS="$ROOT" NI26_ROOT="$ROOT" NI26_DEVICE=cpu NI26_CACHE=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_data1/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I "$HERE/collect_guarded.py" --check-imports
"$PY" -B -I "$HERE/collect_guarded.py"
