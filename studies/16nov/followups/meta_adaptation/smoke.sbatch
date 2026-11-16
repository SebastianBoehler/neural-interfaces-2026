#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani,2080-galvani,a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-s307-smoke
#SBATCH --output=logs/s307-smoke-%j.out
#SBATCH --error=logs/s307-smoke-%j.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/smoke
sha256sum -c source.sha256
sha256sum -c pinned.sha256
(cd /mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116 && sha256sum -c "$OLDPWD/cache.sha256")
export S307_CACHE=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
export S307_CACHE_LOCK_SHA=21c1af2e0784329da2f662319ecdac3f6757715cbcece604fa1db46d21b659bf
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NI26_WS="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/meta_adaptation/crossfit.py P01 P02 smoke
"$PY" -B -I studies/16nov/followups/meta_adaptation/meta_train.py P01 scalar 0 smoke
"$PY" -B -I studies/16nov/followups/meta_adaptation/meta_train.py P01 blocks 0 smoke
