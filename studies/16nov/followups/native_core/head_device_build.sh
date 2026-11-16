#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:?}
cxx=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
torch_root=$($py -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$($py -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
test -f "$torch_root/lib/libtorch_cuda.so"
"$cxx" -std=c++20 -O2 -fPIC -shared "-D_GLIBCXX_USE_CXX11_ABI=$abi" \
    -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include" \
    "$here/head.cpp" -L"$torch_root/lib" -Wl,-rpath,"$torch_root/lib" \
    -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 \
    -Wl,--as-needed -o "${S309_LIBRARY:?}"
