#!/usr/bin/env bash
set -euo pipefail
export PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export S313_DATA=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116/data/proteus_windows.npz
unset PYTHONPATH
printf '%s  %s\n' 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327 "$S313_DATA" | sha256sum -c - >/dev/null
