#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-native353-first-window
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
task_base=/mnt/lustre/work/gehler/gfh527
task_repo="$task_base/neural-interfaces-2026"
task_python="$task_repo/.venv/bin/python"
task_torch="$task_repo/.venv/lib/python3.12/site-packages/torch"
task_cxx="$task_base/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++"
task_source="$task_base/ni26_native353_parity_v3_20261117"
task_assets="$task_base/ni26_native353_qualify_v1_20261117/package"
task_fixture="$task_base/ni26_native353_neural_reference_v1_20261117/reference/fixture.pt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$task_base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$task_source/build/libni26_runtime.so" > worker_ldd.txt
if grep -q 'not found' worker_ldd.txt; then cat worker_ldd.txt; exit 1; fi
awk '$2=="=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' worker_ldd.txt | sort -u > worker_libraries.txt
awk '$0 ~ /^\/mnt\/lustre\//' worker_libraries.txt > worker_nfs_libraries.txt
awk '{print $2}' libraries.sha256 | sort -u > pinned_nfs_libraries.txt
comm -23 worker_nfs_libraries.txt pinned_nfs_libraries.txt > unexpected_nfs_libraries.txt
if test -s unexpected_nfs_libraries.txt; then cat unexpected_nfs_libraries.txt; exit 1; fi
xargs sha256sum < worker_libraries.txt > worker_libraries.sha256
sha256sum -c worker_libraries.sha256 >/dev/null
uname -a > worker_system.txt
task_abi=$("$task_python" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
"$task_cxx" -std=c++20 -O2 -D_GLIBCXX_USE_CXX11_ABI="$task_abi" -I "$task_source/include" \
 -isystem "$task_torch/include" -isystem "$task_torch/include/torch/csrc/api/include" \
 neural_first_window.cpp -L "$task_source/build" -L "$task_torch/lib" \
 -Wl,-rpath,"$task_source/build:$task_torch/lib" -lni26_runtime -lni26_base -ltorch -ltorch_cpu -lc10 -o first_window > build.log 2>&1
./first_window "$task_assets/native.pt" "$task_fixture" "$task_assets" native.pt > native.log 2>&1
"$task_python" neural_first_window.py --package "$task_assets" --fixture "$task_fixture" --native native.pt --project "$task_repo" --out diagnostic.json > diagnostic.log 2>&1
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
sha256sum -c worker_libraries.sha256 >/dev/null
