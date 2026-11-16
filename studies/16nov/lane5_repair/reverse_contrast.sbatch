#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-303-exploratory
#SBATCH --output=logs/reverse-%j.out
#SBATCH --error=logs/reverse-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export ANALYSIS_ROOT="$PWD" NI26_WS=/mnt/lustre/work/gehler/gfh527/ni26_s303_repair_precision_20261116
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: "${ANALYSIS_SHA:?}"
test "$(sha256sum analysis.sha256 | cut -d' ' -f1)" = "$ANALYSIS_SHA"
sha256sum -c analysis.sha256 >/dev/null
(cd "$NI26_WS" && sha256sum -c repair.sha256 >/dev/null && sha256sum -c score.sha256 >/dev/null)
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I reverse_contrast.py
sha256sum -c analysis.sha256 >/dev/null
