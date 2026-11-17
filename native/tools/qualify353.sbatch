#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-native353-components
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
task_base=/mnt/lustre/work/gehler/gfh527
task_repo="$task_base/neural-interfaces-2026"
task_source="$task_base/ni26_native353_base_v2_20261117"
task_python="$task_repo/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$task_base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sha256sum -c source.sha256 >/dev/null
sha256sum "$task_source/build/ni26_probe" "$task_source/build/libni26_base.so" > native_binaries.sha256
"$task_python" "$task_source/tools/convert_package.py" --package "$task_base/ni26_native353_assets_v1_20261117/v66_cp1_cap250.zip" --output package --dataset "$task_repo/data/proteus_windows.npz" --project-root "$task_repo"
"$task_source/build/ni26_probe" package/native.pt package/source_fixture.pt component_output.pt
"$task_python" "$task_source/tests/check_probe.py" package/source_fixture.pt component_output.pt
sha256sum -c native_binaries.sha256
sha256sum -c source.sha256 >/dev/null
