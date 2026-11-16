#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-306-style
#SBATCH --output=logs/style-%j.out
#SBATCH --error=logs/style-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export STYLE_ROOT="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export STYLE_DATA=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116/data/proteus_windows.npz
: "${STYLE_MANIFEST_SHA:?}"
test "$(sha256sum style.sha256 | cut -d' ' -f1)" = "$STYLE_MANIFEST_SHA"
sha256sum -c style.sha256 >/dev/null
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I style.py "${1:?smoke or full}"
sha256sum -c style.sha256 >/dev/null
