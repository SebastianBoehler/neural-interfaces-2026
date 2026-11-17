#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-native353-input-arithmetic
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
task_base=/mnt/lustre/work/gehler/gfh527
task_repo="$task_base/neural-interfaces-2026"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$task_base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uname -a > worker_system.txt
while IFS= read -r path; do sha256sum "$path"; done < system_libraries.txt > worker_system_libraries.sha256
"$task_repo/.venv/bin/python" neural_input_arithmetic.py \
 --package "$task_base/ni26_native353_qualify_v1_20261117/package" \
 --fixture "$task_base/ni26_native353_neural_reference_v1_20261117/reference/fixture.pt" \
 --native "$task_base/ni26_native353_first_window_v3_20261117/native.pt" \
 --project "$task_repo" --out result.json > arithmetic.log 2>&1
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
sha256sum -c worker_system_libraries.sha256 >/dev/null
