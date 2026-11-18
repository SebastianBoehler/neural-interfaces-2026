#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-fat-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:06:00
#SBATCH --job-name=ni26-causal-ft-smoke
#SBATCH --output=refit-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
HERE=$BASE/ni26_friday_causal_rebuild_20261118/refit_full0_smoke_v1
cd "$HERE"
sha256sum --quiet -c source.sha256
(cd "$REPO" && sha256sum --quiet -c "$HERE/refit_inputs.sha256")
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
exec "$REPO/.venv/bin/python" -B refit_smoke.py
