#!/usr/bin/env bash
set -euo pipefail
export PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export S313_BASE=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
export S313_FEATURES=/mnt/lustre/work/gehler/gfh527/ni26_s313_spectral_prepare_v1_20261117/runs/s313_features
export S313_FEATURE_SOURCE_SHA=76c6df2289fa6da9ac363b3e6a7036676488194b08c925520622d12d32b6f3e1
export S313_DATA_SHA=2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327
export S313_SUPPORT=/mnt/lustre/work/gehler/gfh527/ni26_s308_cpu_production_v2_20261116
export S313_KERNEL=/mnt/lustre/work/gehler/gfh527/ni26_s311_cpu_kernel_v1_20261116
export S313_REFERENCES=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export S313_SOLVER_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s309_device_20261116/runs/s309_build/libhead_device.so
export S313_CHECK_LIB="$S313_KERNEL/runs/build/libheadkernels.so"
export S313_OUT="$PWD/runs/s313" S313_MANIFEST="$PWD/source.sha256"
unset PYTHONPATH
printf '%s  %s\n' 4bcbb983086f690780c5690b25fcf2f429a88ff3377d108a7f4b12535fa0fb56 "$NATIVE_CORE_LIB" | sha256sum -c - >/dev/null
printf '%s  %s\n' 1508f45481b8a396d181b99e07cd72206383e51b8674e876cacadc5dbae131de "$NATIVE_TRAIN_LIB" | sha256sum -c - >/dev/null
printf '%s  %s\n' 027df0434ed7096b6110d0c500bbd83c796a473d21426f3953dbc245b11b394e "$S313_SOLVER_LIB" | sha256sum -c - >/dev/null
printf '%s  %s\n' bf3298eb03bbfdfe32b84629ae27e54b77b7c5f0088886b42b11400564805bb0 "$S313_CHECK_LIB" | sha256sum -c - >/dev/null
printf '%s  %s\n' 21c1af2e0784329da2f662319ecdac3f6757715cbcece604fa1db46d21b659bf "$S313_BASE/producer.sha256" | sha256sum -c - >/dev/null
(cd "$S313_BASE" && sha256sum -c producer.sha256 >/dev/null && sha256sum -c source.sha256 >/dev/null)
printf '%s  %s\n' 2eca823bd1f9f181f59255344f164cfd060f363db1f0de18a72d916026b080c5 "$S313_SUPPORT/source.sha256" | sha256sum -c - >/dev/null
(cd "$S313_SUPPORT" && sha256sum -c source.sha256 >/dev/null)
printf '%s  %s\n' 71ee9a414844cce01f70b3d355d15c92c9a10fcb01e899c4ebe77d196f51e14e "$S313_REFERENCES/manifest.json" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$S313_DATA_SHA" "$S313_BASE/data/proteus_windows.npz" | sha256sum -c - >/dev/null
printf '%s  %s\n' 6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f "$S313_BASE/runs/shuffle_perms.npz" | sha256sum -c - >/dev/null
