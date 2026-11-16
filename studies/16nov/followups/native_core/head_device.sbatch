#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=ni26-s309-device
#SBATCH --output=logs/device-%j.out
#SBATCH --error=logs/device-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export S309_PRODUCER=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
export S309_REFERENCES=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all
export S309_LIBRARY="$PWD/runs/s309_build/libhead_device.so"
unset PYTHONPATH
sha256sum -c source.sha256
sha256sum -c inputs.sha256
if [[ "$1" == smoke ]]; then
    test ! -e runs/s309_build
    mkdir -p runs/s309_build
    export PYTHON="$py"
    bash studies/16nov/followups/native_core/head_device_build.sh
    sha256sum "$S309_LIBRARY" > runs/s309_build/library.sha256
else
    sha256sum -c runs/s309_build/library.sha256
fi
srun --cpu-bind=cores "$py" -B -I studies/16nov/followups/native_core/head_device_bench.py "$1"
sha256sum -c source.sha256
sha256sum -c inputs.sha256
