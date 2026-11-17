#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:07:00
#SBATCH --job-name=ni26-native360-smoke
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
suite="${1:?smoke or full required}"
[[ "$suite" == smoke || "$suite" == full ]]
if [[ "$suite" == smoke ]]; then
    modes=(normal stress)
else
    [[ "${2:?normal or stress required}" == normal || "$2" == stress ]]
    modes=("$2")
fi
base=/mnt/lustre/work/gehler/gfh527
repo="$base/neural-interfaces-2026"
build="$base/ni26_native360_package_v1_20261117"
reference="$base/ni26_native353_qualify_v1_20261117/package"
site="$repo/.venv/lib/python3.12/site-packages"
gcc="$base/flightrl-puffer5.pmbvox/toolchain"
image="$repo/runs/cloud/ni26_worker.sif"
check() {
    sha256sum -c source.sha256 >/dev/null
    sha256sum -c inputs.sha256 >/dev/null
    (cd "$build" && sha256sum -c package.sha256 >/dev/null)
}
check
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$gcc/lib"
abi=$("$repo/.venv/bin/python" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
"$gcc/bin/x86_64-conda-linux-gnu-g++" -O2 -std=c++20 -shared -fPIC \
    -D_GLIBCXX_USE_CXX11_ABI="$abi" -I"$build/native/include" \
    -isystem "$site/torch/include" -isystem "$site/torch/include/torch/csrc/api/include" \
    activity.cpp -L"$build/package" -lni26_runtime -o activity.so
if [[ " ${modes[*]} " == *" stress "* ]]; then
    "$repo/.venv/bin/python" stress_assets.py "$reference" "$build/package"
    find stress_reference stress_native -type f -print0 | sort -z | xargs -0 sha256sum > stress.sha256
fi
unset LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > gpu.txt
for mode in "${modes[@]}"; do
    arms=(reference native)
    if [[ "$mode" == stress ]]; then arms=(native reference); fi
    for arm in "${arms[@]}"; do
        package="$reference"
        if [[ "$arm" == native ]]; then package="$build/package"; fi
        if [[ "$mode" == stress ]]; then package="$PWD/stress_$arm"; fi
        apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
            --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
            --bind "$PWD:/work:rw,$repo/2026-competition:/official:ro,$package:/package:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
            --pwd /work "$image" python replay.py "$suite" "$mode" "/work/${arm}_${mode}.json" \
            > "${arm}_${mode}.log" 2>&1
        if grep -E '^\[(own-stage|v6|cpers|count-filter)\].*(error at window|errors [1-9])' "${arm}_${mode}.log"; then
            exit 1
        fi
    done
done
if [[ -e stress.sha256 ]]; then sha256sum -c stress.sha256 >/dev/null; fi
check
python3 compare.py "$suite" "${modes[@]}"
