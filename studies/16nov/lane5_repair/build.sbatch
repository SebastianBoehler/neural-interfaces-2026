#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export CXX=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
export REPAIR_BUILD_DIR="$PWD/runs/build"
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/usr/bin/time -f 'build_seconds=%e peak_rss_kib=%M' bash studies/16nov/lane5_repair/build.sh
cp frozen/liblane5.so "$REPAIR_BUILD_DIR/liblane5.so"
ldd "$REPAIR_BUILD_DIR/liblane5_repair.so" > "$REPAIR_BUILD_DIR/ldd.txt"
if grep -q 'not found' "$REPAIR_BUILD_DIR/ldd.txt"; then cat "$REPAIR_BUILD_DIR/ldd.txt"; exit 1; fi
sha256sum runs/build/liblane5_repair.so > runs/binary.sha256
cat runs/binary.sha256
