#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:03:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-native-init
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
task_root=/mnt/lustre/work/gehler/gfh527
task_python="$task_root/neural-interfaces-2026/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$task_root/ni26_core_car_repair_v1_20261117/build:$task_root/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$task_python" native/tests/initialization_build.py > build.log 2>&1
ldd build/ni26_initialization_source > worker_ldd.txt
if grep -q 'not found' worker_ldd.txt; then exit 1; fi
awk '{for(i=1;i<=NF;i++) if($i ~ /^\//) print $i}' worker_ldd.txt | sort -u | xargs sha256sum > worker_libraries.sha256
finish() {
  status=$?
  trap - EXIT
  sha256sum -c source.sha256 >/dev/null || status=1
  sha256sum -c libraries.sha256 >/dev/null || status=1
  sha256sum -c worker_libraries.sha256 >/dev/null || status=1
  printf 'FINAL_EXIT=%s\n' "$status"
  exit "$status"
}
trap finish EXIT
/usr/bin/time -v "$task_python" native/tests/initialization_fixture.py --project project --output original.pt > original.log 2> original.time
/usr/bin/time -v build/ni26_initialization_source original.pt > native.log 2> native.time
