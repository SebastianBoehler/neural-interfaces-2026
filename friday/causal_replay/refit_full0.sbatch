#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-fat-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:18:00
#SBATCH --job-name=ni26-causal-full0
#SBATCH --output=refit-%A_%a.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
HERE=${SLURM_SUBMIT_DIR:?}
cd "$HERE"
sha256sum --quiet -c source.sha256
(cd "$REPO" && sha256sum --quiet -c "$HERE/refit_inputs.sha256")
(cd "$BASE/ni26_s318_coral_fbl_20261117/runs/diag/ttt/probs18" && sha256sum --quiet -c "$HERE/refit_members.sha256")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
exec "$REPO/.venv/bin/python" -B refit_full0.py "${SLURM_ARRAY_TASK_ID:?}" "$@"
