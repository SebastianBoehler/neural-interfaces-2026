#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-core-source-fit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
base=/mnt/lustre/work/gehler/gfh527/ni26_core_car_repair_v1_20261117/build/libni26_base.so
torch=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/lib/python3.12/site-packages/torch
blas=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/lib/python3.12/site-packages/scipy.libs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$(dirname "$base"):$torch/lib:$blas:/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$base" > worker_ldd.txt
awk '{ for(i=1;i<=NF;i++) if($i ~ /^\//) print $i }' worker_ldd.txt | sort -u | xargs sha256sum > worker_libraries.sha256
finish() {
  status=$?
  trap - EXIT
  sha256sum -c source.sha256 >/dev/null || status=1
  sha256sum -c assets.sha256 >/dev/null || status=1
  sha256sum -c worker_libraries.sha256 >/dev/null || status=1
  printf 'FINAL_EXIT=%s\n' "$status"
  exit "$status"
}
trap finish EXIT
/usr/bin/time -v "$py" -B native/tests/core_training_build.py > build.log 2> build.time
/usr/bin/time -v "$py" -B native/tests/core_training_fixture.py fixture > oracle.log 2> oracle.time
/usr/bin/time -v build/ni26_core_training_source fixture/source.pt fixture/recipe.pt fixture/reference.pt native_core.pt > training.log 2> training.time
