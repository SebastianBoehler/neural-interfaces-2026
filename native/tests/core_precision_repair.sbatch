#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-core-CAR-repair
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
parent=/mnt/lustre/work/gehler/gfh527/ni26_native353_parity_v3_20261117
package=/mnt/lustre/work/gehler/gfh527/ni26_native353_qualify_v1_20261117/package
torch=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/lib/python3.12/site-packages/torch
blas=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/lib/python3.12/site-packages/scipy.libs/libscipy_openblas-5f890258.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$PWD/build:$parent/build:$torch/lib:$(dirname "$blas"):/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$parent/build/libni26_base.so" > worker_ldd.txt
awk '{ for(i=1;i<=NF;i++) if($i ~ /^\//) print $i }' worker_ldd.txt | sort -u | xargs sha256sum > worker_libraries.sha256
"$py" -B core_precision_check.py car "$package/source_fixture.pt" numpy_car.pt
"$py" -B core_precision_build.py
build/core_precision_car "$package/source_fixture.pt" numpy_car.pt
build/ni26_probe "$package/native.pt" "$package/source_fixture.pt" component_output.pt
"$py" -B core_precision_check.py check "$package/source_fixture.pt" result.json --native component_output.pt
sha256sum -c source.sha256 >/dev/null
sha256sum -c worker_libraries.sha256 >/dev/null
