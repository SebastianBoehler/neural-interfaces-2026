#!/bin/bash
# Build only against the exact verified §293 binary and its matching C++ layout.
set -euo pipefail
: "${PYTHON:?}" "${CXX:?}" "${REPAIR_BUILD_DIR:?}"
root=$(cd "$(dirname "$0")/../../.." && pwd)
lib="$root/frozen/liblane5.so"
header="$root/frozen/internal.h"
test "$(sha256sum "$lib" | cut -d' ' -f1)" = adafe96ae57d6b22e9ab396b3f701a23a4ee72c8e38807d842c8089822b0a2c0
test "$(sha256sum "$header" | cut -d' ' -f1)" = 25ec7d8628c5e64b7b7896d57b01f2e555168b733ca58d91c71a020dc1ff9612
test ! -e "$REPAIR_BUILD_DIR"
mkdir "$REPAIR_BUILD_DIR"
torch_root=$($PYTHON -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent)')
abi=$($PYTHON -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
"$CXX" -std=c++20 -O2 -fPIC -shared "-D_GLIBCXX_USE_CXX11_ABI=$abi" \
  -I"$root/frozen" -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include" \
  "$root/studies/16nov/lane5_repair/native.cpp" -L"$root/frozen" -L"$torch_root/lib" \
  -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$torch_root/lib" -Wl,--no-as-needed \
  -llane5 -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed \
  -o "$REPAIR_BUILD_DIR/liblane5_repair.so"
printf 'built %s\n' "$REPAIR_BUILD_DIR/liblane5_repair.so"
