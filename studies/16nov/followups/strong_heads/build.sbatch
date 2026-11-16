#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s308-build
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export CXX=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
sha256sum -c source.sha256
export S308_BUILD_DIR="$PWD/runs/build"
bash studies/16nov/followups/strong_heads/build.sh
export S308_LIB="$S308_BUILD_DIR/libstrongheads.so" S308_TEST_DEVICES=0
"$PYTHON" -B -I -c 'import pathlib,sys,tempfile; sys.path.insert(0,"studies/16nov/followups/strong_heads"); from test_native import test_native_checkpoint_and_stateless_batch; p=tempfile.mkdtemp(); test_native_checkpoint_and_stateless_batch(pathlib.Path(p))'
sha256sum "$S308_LIB" > "$S308_BUILD_DIR/library.sha256"
sha256sum -c source.sha256 >/dev/null
