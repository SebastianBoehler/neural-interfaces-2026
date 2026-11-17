#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-native353-composed-source-update
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c assets.sha256 >/dev/null
sha256sum -c libraries.sha256 >/dev/null
sha256sum -c objects.sha256 >/dev/null
base=/mnt/lustre/work/gehler/gfh527
python="$base/neural-interfaces-2026/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$python" neural_composed_build.py > build.log 2>&1
awk '$2=="=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' worker_ldd.txt | sort -u | xargs sha256sum > worker_libraries.sha256
finish() {
 status=$?
 trap - EXIT
 for manifest in source.sha256 assets.sha256 libraries.sha256 objects.sha256 worker_libraries.sha256; do
  sha256sum -c "$manifest" >/dev/null || status=1
 done
 echo "FINAL_EXIT=$status"
 exit "$status"
}
trap finish EXIT
/usr/bin/time -v "$python" neural_training_fixture.py --out oracle > original.log 2> original.time
/usr/bin/time -v build/neural_training_source "$base/ni26_native353_preparation_v1_20261117/source.pt" oracle/training_fixture.pt > native.log 2> native.time
