#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export CXX=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
export LANE5_BUILD_DIR="$PWD/runs/build"
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ ! -e "$LANE5_BUILD_DIR" ]
[ ! -e "$PWD/runs/binary.sha256" ]
sha256sum -c source.sha256
/usr/bin/time -f 'build_wall_seconds=%e peak_rss_kib=%M' bash studies/16nov/lane5/native/build.sh
[ -f "$LANE5_BUILD_DIR/liblane5.so" ]
ldd "$LANE5_BUILD_DIR/liblane5.so" > "$LANE5_BUILD_DIR/ldd.txt"
if grep -q 'not found' "$LANE5_BUILD_DIR/ldd.txt"; then cat "$LANE5_BUILD_DIR/ldd.txt"; exit 1; fi
readelf -d "$LANE5_BUILD_DIR/liblane5.so" | grep -E 'RPATH|RUNPATH' || true
sha256sum runs/build/liblane5.so > runs/binary.sha256
cat runs/binary.sha256
