#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-native353-FT-stage-diagnostic
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
base=/mnt/lustre/work/gehler/gfh527
repo="$base/neural-interfaces-2026"
python="$repo/.venv/bin/python"
current="$base/ni26_native353_row_mean_repair_v1_20261117"
assets="$base/ni26_native353_qualify_v1_20261117/package"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$python" finetune_diagnostic_build.py > build.log 2>&1
awk '$2=="=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' worker_ldd.txt | sort -u | xargs sha256sum > worker_libraries.sha256
numeric_status=0
build/neural_collect "$assets/native.pt" "$current/source_with_row_means.pt" "$assets" > all_source.log 2>&1 || numeric_status=$?
echo "$numeric_status" > numeric_exit_status.txt
grep -q '^ALL_SOURCE_NUMERIC_FIELDS_COLLECTED$' all_source.log
build/raw_refit_trace "$assets/native.pt" "$current/source_with_row_means.pt" "$assets" native_raw_trace.pt > native_trace.log 2>&1
"$python" finetune_trace.py --package "$assets" --fixture "$current/source_with_row_means.pt" --native native_raw_trace.pt --production neural_observed.pt --project "$repo" --out paired_trace.json > python_trace.log 2>&1
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
sha256sum -c worker_libraries.sha256 >/dev/null
