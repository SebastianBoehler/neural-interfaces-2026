#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-s327-pilot
#SBATCH --output=logs/pilot-%j.out
#SBATCH --error=logs/pilot-%j.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s327_conditional_head_20261117
old=/mnt/lustre/work/gehler/gfh527/ni26_s326_conditional_head_20261117
sha256sum -c source.sha256 >/dev/null
(cd "$old" && sha256sum -c source.sha256 >/dev/null)
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export S309_LIBRARY=/mnt/lustre/work/gehler/gfh527/ni26_s309_device_20261116/runs/s309_build/libhead_device.so
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
printf '%s  %s\n' 027df0434ed7096b6110d0c500bbd83c796a473d21426f3953dbc245b11b394e "$S309_LIBRARY" | sha256sum -c - >/dev/null
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/17nov/conditional_head327/fit.py
"$PY" -B -I studies/17nov/conditional_head327/score.py
