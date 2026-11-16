#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-s314-fit
#SBATCH --output=logs/s314-fit-%A_%a.out
#SBATCH --error=logs/s314-fit-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c s314-source.sha256 >/dev/null
export NI26_WS="$PWD" OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if [[ ${1:-} == smoke ]]; then task=0; extra=(smoke); else task=${SLURM_ARRAY_TASK_ID:?}; extra=(); fi
srun --cpu-bind=cores "$PY" -B -I studies/17nov/time_patterns/produce.py "$task" "${extra[@]}"
