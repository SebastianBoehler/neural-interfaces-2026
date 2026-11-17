#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-native353-integrated
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
task_base=/mnt/lustre/work/gehler/gfh527
task_repo="$task_base/neural-interfaces-2026"
task_python="$task_repo/.venv/bin/python"
task_torch="$task_repo/.venv/lib/python3.12/site-packages/torch"
task_cxx="$task_base/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++"
task_assets="$task_base/ni26_native353_qualify_v1_20261117/package"
task_blas="$task_repo/.venv/lib/python3.12/site-packages/scipy.libs/libscipy_openblas-5f890258.so"
test -r "$task_assets/source_fixture.pt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$task_base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
task_abi=$("$task_python" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
cmake -S . -B build -DCMAKE_CXX_COMPILER="$task_cxx" -DTORCH_ROOT="$task_torch" -DTORCH_CXX11_ABI="$task_abi" -DNI26_LBFGSB_BLAS_LIBRARY="$task_blas" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
build/ni26_probe "$task_assets/native.pt" "$task_assets/source_fixture.pt" component_output.pt
"$task_python" tests/check_probe.py "$task_assets/source_fixture.pt" component_output.pt
"$task_python" tests/linear_fixture.py linear_fixture.pt
build/ni26_linear_source linear_fixture.pt linear_output.pt
sha256sum -c source.sha256 >/dev/null
