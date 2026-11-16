#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=ni26-s308-smoke
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export S308_BASE=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
export S308_CACHE="$S308_BASE/runs/s301_native_inputs"
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export S308_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s308_strongheads_build_v1_20261116/runs/build/libstrongheads.so
export S308_SMOKE_OUT="$PWD/runs/smoke"
export S308_DEVICE=1
unset PYTHONPATH
sha256sum -c source.sha256
( cd /mnt/lustre/work/gehler/gfh527/ni26_s308_strongheads_build_v1_20261116 && sha256sum -c runs/build/library.sha256 )
printf '%s  %s\n' 4bcbb983086f690780c5690b25fcf2f429a88ff3377d108a7f4b12535fa0fb56 "$NATIVE_CORE_LIB" | sha256sum -c -
printf '%s  %s\n' 1508f45481b8a396d181b99e07cd72206383e51b8674e876cacadc5dbae131de "$NATIVE_TRAIN_LIB" | sha256sum -c -
printf '%s  %s\n' 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327 "$S308_BASE/data/proteus_windows.npz" | sha256sum -c -
printf '%s  %s\n' 6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f "$S308_BASE/runs/shuffle_perms.npz" | sha256sum -c -
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON" -B -I studies/16nov/followups/strong_heads/smoke_real.py
sha256sum -c source.sha256 >/dev/null
