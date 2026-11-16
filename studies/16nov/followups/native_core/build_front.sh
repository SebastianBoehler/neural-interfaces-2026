#!/usr/bin/env bash
# Build the front end alone for its no-score parity gate.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python3}
root=$($py -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$($py -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
out=${NATIVE_CORE_BUILD_DIR:?set a private build directory}
mkdir -p "$out"
args=(-std=c++20 -O2 -fPIC "-D_GLIBCXX_USE_CXX11_ABI=$abi" -I"$root/include" -I"$root/include/torch/csrc/api/include")
objects=()
for name in filter preprocess covariance tangent; do
    "${CXX:-c++}" "${args[@]}" -c "$here/$name.cpp" -o "$out/$name.o"
    objects+=("$out/$name.o")
done
libs=(-L"$root/lib" -Wl,-rpath,"$root/lib" -ltorch -ltorch_cpu -lc10)
if [[ -f "$root/lib/libtorch_cuda.so" ]]; then
    libs=(-L"$root/lib" -Wl,-rpath,"$root/lib" -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed)
fi
"${CXX:-c++}" -shared "${objects[@]}" "${libs[@]}" -o "$out/libnative_core_front.so"
