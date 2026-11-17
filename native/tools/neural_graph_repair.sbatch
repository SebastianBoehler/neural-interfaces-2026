#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-native353-source-graph-repair
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
for manifest in source.sha256 assets.sha256 libraries.sha256 objects.sha256; do sha256sum -c "$manifest" >/dev/null; done
base=/mnt/lustre/work/gehler/gfh527
python="$base/neural-interfaces-2026/.venv/bin/python"
original="$base/ni26_native353_training_composed_v1_20261117"
prep="$base/ni26_native353_preparation_v1_20261117/source.pt"
fixture="$original/oracle/training_fixture.pt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$python" neural_graph_repair_build.py > build.log 2>&1
awk '$2=="=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' worker_ldd.txt | sort -u | xargs sha256sum > worker_libraries.sha256
finish() {
 status=$?
 trap - EXIT
 for manifest in source.sha256 assets.sha256 libraries.sha256 objects.sha256 worker_libraries.sha256; do sha256sum -c "$manifest" >/dev/null || status=1; done
 echo "FINAL_EXIT=$status"
 exit "$status"
}
trap finish EXIT
status=0
"$original/build/neural_training_source" "$prep" "$fixture" > reproduction.log 2> reproduction.stderr || status=$?
test "$status" -eq 1
grep -q '^source training oracle mismatch: state/spatial_conv.0.bias$' reproduction.stderr
if cmp -s reproduction.log "$original/native.log"; then echo 'OLD_REPRODUCTION_STDOUT_BYTE_EXACT=1'; else echo 'OLD_REPRODUCTION_STDOUT_BYTE_EXACT=0'; fi
/usr/bin/time -v build/neural_training_source "$prep" "$fixture" > native.log 2> native.time
