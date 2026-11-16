#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s301-build
#SBATCH --output=logs/front-build-%j.out
#SBATCH --error=logs/front-build-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum source.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export CXX=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONPATH
sha256sum -c front.sha256
test ! -e runs/s301_build
export NATIVE_CORE_BUILD_DIR="$PWD/runs/s301_build"
bash studies/16nov/followups/native_core/build_front.sh
export NATIVE_CORE_LIB="$NATIVE_CORE_BUILD_DIR/libnative_core_front.so" NATIVE_CORE_TEST_DEVICES=0
"$PYTHON" -B -I studies/16nov/followups/native_core/smoke_front.py
sha256sum "$NATIVE_CORE_LIB" > "$NATIVE_CORE_BUILD_DIR/library.sha256"
sha256sum -c front.sha256 >/dev/null
