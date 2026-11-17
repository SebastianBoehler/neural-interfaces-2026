#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-native353-package
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c inputs.sha256 >/dev/null
base=/mnt/lustre/work/gehler/gfh527
venv="$base/neural-interfaces-2026/.venv"
site="$venv/lib/python3.12/site-packages"
gcc="$base/flightrl-puffer5.pmbvox/toolchain"
models="$base/ni26_native353_qualify_v1_20261117/package"
blas="$site/scipy.libs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$gcc/lib"
abi=$("$venv/bin/python" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$gcc/bin/x86_64-conda-linux-gnu-g++" \
    -DCMAKE_C_COMPILER="$gcc/bin/x86_64-conda-linux-gnu-gcc" \
    -DTORCH_ROOT="$site/torch" -DTORCH_CXX11_ABI="$abi" \
    -DNI26_LBFGSB_BLAS_LIBRARY="$blas/libscipy_openblas-5f890258.so" \
    -DCUDA_INCLUDE_DIR="$site/triton/backends/nvidia/include" > configure.log 2>&1
cmake --build build --target ni26_runtime -j "$SLURM_CPUS_PER_TASK" > build.log 2>&1
cmake --install build --prefix "$PWD/package"
cp "$models"/{native,eye_s*,rest_s*,sbn_*,teacher}.pt package/

# The Galvani compiler adds its own search directory to the linked ELF.
cat > normalize_rpath.cmake <<'CMAKE'
file(RPATH_CHANGE FILE "${LIBRARY}" OLD_RPATH "${OLD_RPATH}" NEW_RPATH "$ORIGIN")
CMAKE
for library in package/libni26_{base,runtime}.so; do
    rpath=$(readelf -d "$library" | sed -nE 's/.*(RPATH|RUNPATH).*\[(.*)\].*/\2/p')
    case "$rpath" in
        '$ORIGIN') ;;
        '$ORIGIN:'"$gcc/lib")
            cmake -DLIBRARY="$library" -DOLD_RPATH="$rpath" -P normalize_rpath.cmake ;;
        *) echo "Unexpected runtime search path in $library" >&2; exit 1 ;;
    esac
    readelf -d "$library" > "$library.elf"
    grep -Fq '[$ORIGIN]' "$library.elf"
    if grep -E '(RPATH|RUNPATH).*mnt/lustre' "$library.elf"; then exit 1; fi
    if grep -E 'NEEDED.*(openblas|gfortran|quadmath)' "$library.elf"; then exit 1; fi
    mv "$library.elf" .
done
sha256sum -c source.sha256 >/dev/null
sha256sum -c inputs.sha256 >/dev/null
sha256sum package/* > package.sha256
echo NATIVE_PACKAGE_INSTALL_PASS
