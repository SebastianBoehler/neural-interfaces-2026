#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-native353-parity
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$task_base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
task_abi=$("$task_python" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
cmake -S . -B build -DCMAKE_CXX_COMPILER="$task_cxx" -DTORCH_ROOT="$task_torch" -DTORCH_CXX11_ABI="$task_abi" -DNI26_LBFGSB_BLAS_LIBRARY="$task_blas" -DCUDA_INCLUDE_DIR="$task_repo/.venv/lib/python3.12/site-packages/triton/backends/nvidia/include" -DCMAKE_BUILD_TYPE=Release > configure.log 2>&1
cmake --build build -j 4 > build.log 2>&1
test_status=0
run_check() {
  local name="$1"
  shift
  if "$@" > "$name.log" 2>&1; then echo "PASS $name"; else echo "FAIL $name"; test_status=1; fi
}
run_check components build/ni26_probe "$task_assets/native.pt" "$task_assets/source_fixture.pt" component_output.pt
if test -r component_output.pt; then run_check component_parity "$task_python" tests/check_probe.py "$task_assets/source_fixture.pt" component_output.pt; fi
run_check routing build/ni26_routing_source "$task_assets/native.pt" "$task_base/ni26_native353_routing_components1/reference/fixture.pt"
"$task_python" tools/hash_fixture.py "$task_assets/sbn_k2_s0.pt" real_fir.bin
run_check hash build/ni26_hash_test real_fir.bin
run_check neural build/ni26_neural_source "$task_assets/native.pt" "$task_base/ni26_native353_neural_reference_v1_20261117/reference/fixture.pt" "$task_assets"
sha256sum -c source.sha256 >/dev/null
exit "$test_status"
