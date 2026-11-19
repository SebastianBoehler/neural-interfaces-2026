#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-runs0-fbraw-canary
#SBATCH --output=fbraw-canary-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
here=${SLURM_SUBMIT_DIR:?}
cd "$here"
sha256sum --quiet -c canary_source.sha256
(cd "$base" && sha256sum --quiet -c "$here/runs0_warm_inputs.sha256")
export PYTHONPATH="$repo/experiments:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
exec "$repo/.venv/bin/python" -B runs0_fbraw_canary.py
