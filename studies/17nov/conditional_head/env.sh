#!/bin/bash
set -euo pipefail
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export S309_LIBRARY=/mnt/lustre/work/gehler/gfh527/ni26_s309_device_20261116/runs/s309_build/libhead_device.so
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
printf '%s  %s\n' 027df0434ed7096b6110d0c500bbd83c796a473d21426f3953dbc245b11b394e "$S309_LIBRARY" | sha256sum -c - >/dev/null
sha256sum -c source.sha256 >/dev/null
"$PY" -B -I -c 'import torch, numpy, sklearn; print("imports passed", flush=True)'
