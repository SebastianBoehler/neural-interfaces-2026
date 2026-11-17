#!/bin/bash
set -euo pipefail
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
cxx=/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/bin/x86_64-conda-linux-gnu-g++
original=/mnt/lustre/work/gehler/gfh527/ni26_s311_cpu_kernel_v1_20261116
out="$PWD/build"
test ! -e "$out"
mkdir "$out"
"$py" generate_validation.py
root=$("$py" -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent)')
abi=$("$py" -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
flags=(-std=c++20 -O2 -fPIC "-D_GLIBCXX_USE_CXX11_ABI=$abi" -I"$root/include" -I"$root/include/torch/csrc/api/include" -I"$original/studies/16nov/followups/native_core")
"$cxx" "${flags[@]}" -c "$out/head_validation.cpp" -o "$out/head.o"
"$cxx" "${flags[@]}" -c "$original/studies/16nov/followups/head_kernel/kernel.cpp" -o "$out/kernel.o"
libs=(-L"$root/lib" -Wl,-rpath,"$root/lib" -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed)
"$cxx" -shared "$out/head.o" "$out/kernel.o" "${libs[@]}" -o "$out/libheadvalidation.so"
sha256sum "$out/libheadvalidation.so" "$out/head_validation.cpp" > "$out/library.sha256"
