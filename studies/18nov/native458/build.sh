#!/bin/bash
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
py=$b/neural-interfaces-2026/.venv/bin/python
tc=$b/flightrl-puffer5.pmbvox/toolchain
export LD_LIBRARY_PATH="$tc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
read -r root abi < <("$py" -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent,int(torch.compiled_with_cxx11_abi()))')
"$tc/bin/x86_64-conda-linux-gnu-g++" -std=c++20 -O1 -shared -fPIC "-D_GLIBCXX_USE_CXX11_ABI=$abi" \
 -I"$root/include" -I"$root/include/torch/csrc/api/include" decoder.cpp -L"$root/lib" -Wl,-rpath,"$root/lib" \
 -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -o decoder.so
sha256sum decoder.so > library.sha256
