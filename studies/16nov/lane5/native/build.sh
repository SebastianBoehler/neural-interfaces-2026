#!/bin/bash
# Build against the selected existing PyTorch install; no downloads or installs.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python3}
cxx=${CXX:-c++}
torch_root=$($py -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$($py -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
out=${LANE5_BUILD_DIR:-"$here/build"}
mkdir -p "$out"
common=(-std=c++20 -O2 -fPIC "-D_GLIBCXX_USE_CXX11_ABI=$abi" -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include")
objects=()
for src in model train session api; do
    "$cxx" "${common[@]}" -c "$here/$src.cpp" -o "$out/$src.o"
    objects+=("$out/$src.o")
done
libs=(-L"$torch_root/lib" -Wl,-rpath,"$torch_root/lib" -ltorch -ltorch_cpu -lc10)
if [[ -f "$torch_root/lib/libtorch_cuda.so" ]]; then
    libs=(-L"$torch_root/lib" -Wl,-rpath,"$torch_root/lib" -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed)
fi
"$cxx" -shared "${objects[@]}" "${libs[@]}" -o "$out/liblane5.so"
printf 'built %s\n' "$out/liblane5.so"
