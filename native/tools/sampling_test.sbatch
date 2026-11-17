#!/bin/bash
#SBATCH --job-name=ni26-native-sampler-exact
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
running=$(squeue --me -h -r -t RUNNING | wc -l)
active=$(squeue --me -h -r | wc -l)
if ((running>32 || active>244)); then exit 1; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
base=/mnt/lustre/work/gehler/gfh527
toolchain="$base/flightrl-puffer5.pmbvox/toolchain"
"$toolchain/bin/x86_64-conda-linux-gnu-g++" -std=c++20 -O2 -I include \
 src/sampling.cpp src/sampling_rng.cpp src/sampling_choice.cpp tests/sampling_source.cpp \
 -Wl,-rpath,"$toolchain/lib" -o sampling_source
"$base/neural-interfaces-2026/.venv/bin/python" tools/sampling_metadata_oracle.py --out oracle > oracle.log
./sampling_source oracle > exact.log
