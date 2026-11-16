#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --job-name=ni26-s301-trainref
#SBATCH --output=logs/trainref-%j.out
#SBATCH --error=logs/trainref-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum trainref.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
sha256sum -c trainref.sha256
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
CXX=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
root=$($PY -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$($PY -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
test ! -e runs/s301_trainref
mkdir -p runs/s301_trainref
"$CXX" -std=c++20 -O2 -shared -fPIC "-D_GLIBCXX_USE_CXX11_ABI=$abi" \
    -I"$root/include" -I"$root/include/torch/csrc/api/include" \
    studies/16nov/followups/native_core/train_ref.cpp -L"$root/lib" -Wl,-rpath,"$root/lib" \
    -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed \
    -o runs/s301_trainref/libtrainref.so
export NATIVE_TRAIN_LIB="$PWD/runs/s301_trainref/libtrainref.so" NATIVE_CORE_TEST_DEVICES=0
"$PY" -B -I studies/16nov/followups/native_core/smoke_trainref.py
sha256sum "$NATIVE_TRAIN_LIB" > runs/s301_trainref/library.sha256
