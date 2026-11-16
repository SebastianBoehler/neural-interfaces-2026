#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=logs/score-%j.out
#SBATCH --error=logs/score-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" LANE5_LIB="$PWD/runs/build/liblane5.so" LANE5_REPAIR_LIB="$PWD/runs/build/liblane5_repair.so"
export REPAIR_MANIFEST="$PWD/repair.sha256" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: "${REPAIR_MANIFEST_SHA:?}" "${REPAIR_SCORE_SHA:?}"
test "$(sha256sum repair.sha256 | cut -d' ' -f1)" = "$REPAIR_MANIFEST_SHA"
test "$(sha256sum score.sha256 | cut -d' ' -f1)" = "$REPAIR_SCORE_SHA"
test "$(cat runs/gates/ready)" = "$REPAIR_MANIFEST_SHA"
sha256sum -c source.sha256 >/dev/null
sha256sum -c repair.sha256 >/dev/null
sha256sum -c score.sha256 >/dev/null
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/lane5_repair/score.py
sha256sum -c repair.sha256 >/dev/null
sha256sum -c score.sha256 >/dev/null
