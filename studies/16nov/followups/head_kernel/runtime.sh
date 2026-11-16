#!/usr/bin/env bash
set -euo pipefail
export PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export CXX=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export S311_BASE=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
export S311_CACHE="$S311_BASE/runs/s301_native_inputs"
export S311_REFERENCES=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export S311_LIB="$PWD/runs/build/libheadkernels.so"
unset PYTHONPATH
printf '%s  %s\n' 21c1af2e0784329da2f662319ecdac3f6757715cbcece604fa1db46d21b659bf "$S311_BASE/producer.sha256" | sha256sum -c - >/dev/null
( cd "$S311_BASE" && sha256sum -c producer.sha256 >/dev/null && sha256sum -c source.sha256 >/dev/null )
printf '%s  %s\n' 71ee9a414844cce01f70b3d355d15c92c9a10fcb01e899c4ebe77d196f51e14e "$S311_REFERENCES/manifest.json" | sha256sum -c - >/dev/null
printf '%s  %s\n' 4bcbb983086f690780c5690b25fcf2f429a88ff3377d108a7f4b12535fa0fb56 "$NATIVE_CORE_LIB" | sha256sum -c - >/dev/null
printf '%s  %s\n' 1508f45481b8a396d181b99e07cd72206383e51b8674e876cacadc5dbae131de "$NATIVE_TRAIN_LIB" | sha256sum -c - >/dev/null
printf '%s  %s\n' 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327 "$S311_BASE/data/proteus_windows.npz" | sha256sum -c - >/dev/null
