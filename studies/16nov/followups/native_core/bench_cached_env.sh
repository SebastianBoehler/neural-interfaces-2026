#!/usr/bin/env bash
set -euo pipefail
export LANE5_MANIFEST_SHA="$(sha256sum cached.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export NI26_ROOT="$PWD" CUDA_VISIBLE_DEVICES=''
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export S301_BENCH_REFERENCE=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_20261116/runs/s301_reference/P01_within0.npz
export S301_CACHED_SMOKE="$PWD/runs/s301_cached_bench/smoke.json" S301_CACHED_OUTPUT="$PWD/runs/s301_cached_bench/results.json"
unset PYTHONPATH
sha256sum -c source.sha256 >/dev/null
sha256sum -c cached.sha256
(cd /mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116 && sha256sum -c full.sha256 >/dev/null && sha256sum -c runs/s301_full_build/library.sha256)
printf 'CPU allocation: '; taskset -pc $$
grep -m1 'model name' /proc/cpuinfo
