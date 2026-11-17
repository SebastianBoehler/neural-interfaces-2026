#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-core-precision
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
cxx=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
parent=/mnt/lustre/work/gehler/gfh527/ni26_native353_parity_v3_20261117
package=/mnt/lustre/work/gehler/gfh527/ni26_native353_qualify_v1_20261117/package
torch=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/lib/python3.12/site-packages/torch
blas=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/lib/python3.12/site-packages/scipy.libs/libscipy_openblas-5f890258.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$parent/build:$torch/lib:$(dirname "$blas"):/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Worker-local OS libraries are recorded after placement, never pinned from the login host.
ldd "$parent/build/libni26_base.so" > worker_ldd.txt
awk '/=> \/|^\// { for(i=1;i<=NF;i++) if($i ~ /^\//) print $i }' worker_ldd.txt | sort -u | xargs sha256sum > worker_libraries.sha256
"$py" -B core_precision.py reference
abi=$("$py" -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
"$cxx" -std=c++20 -O2 "-D_GLIBCXX_USE_CXX11_ABI=$abi" -I"$parent/include" -I"$torch/include" -I"$torch/include/torch/csrc/api/include" core_precision_dump.cpp \
  -L"$parent/build" -L"$torch/lib" -lni26_base -ltorch -ltorch_cpu -lc10 -ltorch_cuda -lc10_cuda "$blas" -o core_precision_dump
./core_precision_dump "$package/native.pt" "$package/source_fixture.pt" reference_stages.pt native_stages.pt initial
"$py" -B core_precision.py prepare_means
./core_precision_dump "$package/native.pt" "$package/source_fixture.pt" mean_oracles.pt mean_stages.pt means
"$py" -B core_precision.py report
sha256sum -c source.sha256 >/dev/null
sha256sum -c worker_libraries.sha256 >/dev/null
