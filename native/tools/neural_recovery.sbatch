#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-native353-neural-recovery
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
running=$(squeue --me -h -r -t RUNNING | wc -l)
active=$(squeue --me -h -r | wc -l)
if ((running>32 || active>244)); then exit 1; fi
task_base=/mnt/lustre/work/gehler/gfh527
task_repo="$task_base/neural-interfaces-2026"
task_python="$task_repo/.venv/bin/python"
task_torch="$task_repo/.venv/lib/python3.12/site-packages/torch"
task_cxx="$task_base/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++"
task_assets="$task_base/ni26_native353_qualify_v1_20261117/package"
task_fixture="$task_base/ni26_native353_neural_reference_v1_20261117/reference/fixture.pt"
task_blas="$task_repo/.venv/lib/python3.12/site-packages/scipy.libs/libscipy_openblas-5f890258.so"
task_cuda="$task_repo/.venv/lib/python3.12/site-packages/triton/backends/nvidia/include"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$task_base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
task_abi=$("$task_python" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
cmake -S . -B build -DCMAKE_CXX_COMPILER="$task_cxx" -DTORCH_ROOT="$task_torch" \
 -DTORCH_CXX11_ABI="$task_abi" -DNI26_LBFGSB_BLAS_LIBRARY="$task_blas" \
 -DCUDA_INCLUDE_DIR="$task_cuda" -DCMAKE_BUILD_TYPE=Release > configure.log 2>&1
cmake --build build -j 4 > build.log 2>&1
"$task_python" tools/hash_fixture.py "$task_assets/sbn_k2_s0.pt" real_fir.bin
build/ni26_hash_test real_fir.bin > hash.log 2>&1
/usr/bin/time -v build/ni26_neural_source "$task_assets/native.pt" "$task_fixture" "$task_assets" > neural.log 2>&1
sha256sum -c assets.sha256 >/dev/null
sha256sum -c source.sha256 >/dev/null
