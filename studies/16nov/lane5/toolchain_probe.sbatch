#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/toolchain-%j.out
#SBATCH --error=logs/toolchain-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
torch_dir="$($py -B -I -c 'import torch; from pathlib import Path; print(Path(torch.__file__).resolve().parent)')"
probe_dir="$PWD/runs/toolchain-$SLURM_JOB_ID"
mkdir "$probe_dir"
cat > "$probe_dir/probe.cpp" <<'CPP'
#include <torch/torch.h>
#include <iostream>
int main() {
    const auto x = torch::ones({2, 3});
    std::cout << "LIBTORCH_CPU_OK elements=" << x.numel() << " sum=" << x.sum().item<float>() << "\n";
    return x.numel() == 6 && x.sum().item<float>() == 6 ? 0 : 1;
}
CPP
printf 'compiler: '; g++ --version | head -1
printf 'cmake: '; cmake --version | head -1
printf 'CUDA headers: '
if [ -f /usr/local/cuda/include/cuda.h ]; then printf 'system present\n'; else printf 'no system CUDA header\n'; fi
/usr/bin/time -f 'compile_wall_seconds=%e peak_rss_kib=%M' g++ -std=c++17 -O0 -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
    "$probe_dir/probe.cpp" -L"$torch_dir/lib" -Wl,-rpath,"$torch_dir/lib" \
    -ltorch -ltorch_cpu -lc10 -o "$probe_dir/probe"
"$probe_dir/probe"
