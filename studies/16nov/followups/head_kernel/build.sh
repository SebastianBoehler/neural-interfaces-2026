#!/usr/bin/env bash
set -euo pipefail
py=${PYTHON:-python3}
cxx=${CXX:-c++}
out=${S311_BUILD_DIR:?must provide a fresh build directory}
test ! -e "$out"
mkdir -p "$out"
root=$($py -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$($py -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
flags=(-std=c++20 -O2 -fPIC "-D_GLIBCXX_USE_CXX11_ABI=$abi" -I"$root/include" -I"$root/include/torch/csrc/api/include")
"$cxx" "${flags[@]}" -c studies/16nov/followups/native_core/head.cpp -o "$out/head.o"
"$cxx" "${flags[@]}" -c studies/16nov/followups/head_kernel/kernel.cpp -o "$out/kernel.o"
libs=(-L"$root/lib" -Wl,-rpath,"$root/lib" -ltorch -ltorch_cpu -lc10)
if [[ -f "$root/lib/libtorch_cuda.so" ]]; then
  libs=(-L"$root/lib" -Wl,-rpath,"$root/lib" -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed)
fi
"$cxx" -shared "$out/head.o" "$out/kernel.o" "${libs[@]}" -o "$out/libheadkernels.so"
