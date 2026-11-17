#!/bin/bash
set -euo pipefail
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
cxx=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
root=$("$py" -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$("$py" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
test ! -e build
mkdir build
flags=(-std=c++20 -O2 -fPIC "-D_GLIBCXX_USE_CXX11_ABI=$abi" -I"$root/include" -I"$root/include/torch/csrc/api/include")
"$cxx" "${flags[@]}" -c span.cpp -o build/span.o
"$cxx" "${flags[@]}" -c reference.cpp -o build/reference.o
"$cxx" "${flags[@]}" -c head.cpp -o build/head.o
"$cxx" -shared build/span.o build/reference.o build/head.o -L"$root/lib" -Wl,-rpath,"$root/lib" -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -o build/libspan.so
sha256sum build/libspan.so > build/library.sha256
