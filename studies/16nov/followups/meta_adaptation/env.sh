export S307_CACHE=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
export S307_CACHE_LOCK_SHA=21c1af2e0784329da2f662319ecdac3f6757715cbcece604fa1db46d21b659bf
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NI26_WS="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
