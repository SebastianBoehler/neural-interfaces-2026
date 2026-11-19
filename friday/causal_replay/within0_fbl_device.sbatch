#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-fbl-device
#SBATCH --output=fbl-device-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
here=${SLURM_SUBMIT_DIR:?}
cd "$here"
sha256sum --quiet -c source.sha256
(cd "$base" && sha256sum --quiet -c "$here/within0_frozen_inputs.sha256")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/experiments:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
device=${1:?cpu or a100}
if [[ "$device" == cpu ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi
exec "$repo/.venv/bin/python" -B within0_fbl_device.py "$device"
