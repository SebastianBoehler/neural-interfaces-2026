#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-source-PCA
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
base=/mnt/lustre/work/gehler/gfh527/ni26_core_car_repair_v1_20261117/build
torch=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/lib/python3.12/site-packages/torch
blas=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/lib/python3.12/site-packages/scipy.libs
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$base:$torch/lib:$blas:/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd build/ni26_pca_source > worker_ldd.txt
grep -F "$base/libni26_base.so" worker_ldd.txt >/dev/null
if grep -q 'not found' worker_ldd.txt; then exit 1; fi
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
/usr/bin/time -v "$py" -B native/tests/pca_fixture.py common.pt > setup.log 2> setup.time
/usr/bin/time -v "$py" -B native/tests/pca_reference.py common.pt build/ni26_pca_source comparison > comparison.log 2> comparison.time
