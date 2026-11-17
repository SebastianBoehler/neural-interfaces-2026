#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-native353-source-prefix
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
for manifest in source assets objects libraries; do sha256sum -c "$manifest.sha256" > /dev/null; done
base=/mnt/lustre/work/gehler/gfh527
project="$base/ni26_raw312_cpu_20261117_v2"
python="$base/neural-interfaces-2026/.venv/bin/python"
fixture="$base/ni26_native353_training_composed_v1_20261117/oracle/training_fixture.pt"
prep="$base/ni26_native353_preparation_v1_20261117/source.pt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

finish() {
  status=$?
  trap - EXIT
  for manifest in source assets objects libraries; do sha256sum -c "$manifest.sha256" > /dev/null || status=1; done
  if test -f worker_libraries.sha256; then sha256sum -c worker_libraries.sha256 > /dev/null || status=1; fi
  echo "FINAL_EXIT=$status"
  exit "$status"
}
trap finish EXIT
"$python" neural_training_prefix_build.py > build.log 2>&1
awk '$2=="=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' worker_ldd.txt | sort -u | xargs sha256sum > worker_libraries.sha256
/usr/bin/time -v "$python" neural_training_prefix.py --fixture "$fixture" --project "$project" --output original_prefix.pt > original.log 2> original.time
/usr/bin/time -v build/neural_training_source "$prep" "$fixture" original_prefix.pt > native.log 2> native.time
