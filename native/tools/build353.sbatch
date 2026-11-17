#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-native353-build
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
task_repo=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
task_python="$task_repo/.venv/bin/python"
task_torch="$task_repo/.venv/lib/python3.12/site-packages/torch"
task_cxx=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
task_package=/mnt/lustre/work/gehler/gfh527/ni26_native353_assets_v1_20261117/v66_cp1_cap250.zip
test -r "$task_package"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
task_abi=$("$task_python" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
cmake -S . -B build -DCMAKE_CXX_COMPILER="$task_cxx" -DTORCH_ROOT="$task_torch" -DTORCH_CXX11_ABI="$task_abi" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
"$task_python" tools/convert_package.py --package "$task_package" --output package --dataset "$task_repo/data/proteus_windows.npz" --project-root "$task_repo"
build/ni26_probe package/native.pt package/source_fixture.pt component_output.pt
"$task_python" tests/check_probe.py package/source_fixture.pt component_output.pt
sha256sum -c source.sha256 >/dev/null
