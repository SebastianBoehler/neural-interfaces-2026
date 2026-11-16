#!/usr/bin/env bash
set -euo pipefail
export PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export CXX=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export S311_OLD=/mnt/lustre/work/gehler/gfh527/ni26_s311_cpu_kernel_v1_20261116
export S311_LIB="$S311_OLD/runs/build/libheadkernels.so"
export S311_BLAS_LIB="$PWD/runs/build/libblashead.so"
unset PYTHONPATH
printf '%s  %s\n' bf3298eb03bbfdfe32b84629ae27e54b77b7c5f0088886b42b11400564805bb0 "$S311_LIB" | sha256sum -c - >/dev/null
printf '%s  %s\n' 59ffcd5096d0b6ffbe071cd571a2caba876fa64eb0123ddd5faa00898cca411b /usr/lib64/libopenblasp.so.0 | sha256sum -c - >/dev/null
printf '%s  %s\n' a38ec018aaf2f727916fa402bb082659543dcfbf758d44e6ee3f9819e4ef8b24 "$S311_OLD/runs/s311_smoke/source_matrix.npz" | sha256sum -c - >/dev/null
