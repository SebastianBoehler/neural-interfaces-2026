#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani,2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-306-experts
#SBATCH --output=logs/expert-%A_%a.out
#SBATCH --error=logs/expert-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export EXPERT_ROOT="$PWD" NI26_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
: "${EXPERT_MANIFEST_SHA:?}"
test "$(sha256sum expert.sha256 | cut -d' ' -f1)" = "$EXPERT_MANIFEST_SHA"
sha256sum -c expert.sha256 >/dev/null
if [[ ${1:-} == smoke ]]; then
  /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I run.py 0 smoke
else
  /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I ready.py
  /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I run.py "${SLURM_ARRAY_TASK_ID:?}" full
fi
sha256sum -c expert.sha256 >/dev/null
