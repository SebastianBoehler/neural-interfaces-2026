#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python3}
cxx=${CXX:-c++}
out=${S308_BUILD_DIR:?must provide a fresh build directory}
test ! -e "$out"
mkdir -p "$out"
root=$($py -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$($py -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
libs=(-L"$root/lib" -Wl,-rpath,"$root/lib" -ltorch -ltorch_cpu -lc10)
if [[ -f "$root/lib/libtorch_cuda.so" ]]; then
  libs=(-L"$root/lib" -Wl,-rpath,"$root/lib" -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed)
fi
"$cxx" -std=c++20 -O2 -fPIC -shared "-D_GLIBCXX_USE_CXX11_ABI=$abi" \
  -I"$root/include" -I"$root/include/torch/csrc/api/include" "$here/native.cpp" \
  "${libs[@]}" -o "$out/libstrongheads.so"
