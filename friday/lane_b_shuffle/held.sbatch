#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:18:00
#SBATCH --job-name=ni26-lane-b-held
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
parent=$base/ni26_friday_causal_rebuild_20261118
original=$base/ni26_s318_coral_fbl_20261117
sha256sum --quiet -c source.sha256
sha256sum --quiet -c held.sha256
grep -Fxq OK replay.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/2026-competition:$PWD"
mode=${1:?}
if [[ $mode == collect ]]; then
    exec "$repo/.venv/bin/python" -B held.py collect "$repo/data/proteus_windows.npz" out/
fi
orders=(full0 within0)
order=${orders[${SLURM_ARRAY_TASK_ID:-0}]}
exec "$repo/.venv/bin/python" -B held.py "$mode" "$order" zips "$repo/data/proteus_windows.npz" "$parent" \
    "$original/runs/shuffle_perms.npz" "out/$order.npz"
