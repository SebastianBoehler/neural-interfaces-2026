#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=ni26-s301-parity
#SBATCH --output=logs/parity-%j.out
#SBATCH --error=logs/parity-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum parity.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export S301_REFERENCE=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_20261116/runs/s301_reference
full=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116
export NATIVE_CORE_LIB="$full/runs/s301_full_build/libnative_core.so"
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONPATH
sha256sum -c parity.sha256
(cd "$full" && sha256sum -c full.sha256 >/dev/null && sha256sum -c runs/s301_full_build/library.sha256)
ref="$(dirname "$S301_REFERENCE")/.."
(cd "$ref" && sha256sum -c reference.sha256 >/dev/null)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/parity.py
