#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-native353-row-mean-parity
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
sha256sum -c objects.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
base=/mnt/lustre/work/gehler/gfh527
repo="$base/neural-interfaces-2026"
python="$repo/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$PWD/build:$base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$python" neural_reduction_build.py > build.log 2>&1
awk '$2=="=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' worker_ldd.txt | sort -u | xargs sha256sum > worker_libraries.sha256
sha256sum -c worker_libraries.sha256 >/dev/null
"$python" neural_row_means.py "$base/ni26_native353_neural_reference_v1_20261117/reference/fixture.pt" source_with_row_means.pt > mean_oracle.json
build/ni26_neural_source "$base/ni26_native353_qualify_v1_20261117/package/native.pt" source_with_row_means.pt "$base/ni26_native353_qualify_v1_20261117/package" > neural.log 2>&1
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
sha256sum -c objects.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
sha256sum -c worker_libraries.sha256 >/dev/null
