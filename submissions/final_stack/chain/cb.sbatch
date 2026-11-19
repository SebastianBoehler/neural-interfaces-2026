#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:50:00
#SBATCH --job-name=ni26-cbbank
#SBATCH --output=cbbank-%A_%a.out
# usage (release dir): sbatch --array=0-2 cb.sbatch ORDER     (array id = seed); streams the LOPO CBraMod-C fits through CBMember
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --check --status source.sha256
repo=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
order=${1:?}
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
bank=()
[[ $order == full0 ]] && bank=(--bank $PWD/ensemble_C_held_new.npz)
exec "$repo/.venv/bin/python" -B -W ignore cb_bank.py "$repo" "$PWD/ckpt" "$PWD/cbbank_${order}_s${SLURM_ARRAY_TASK_ID}.npz" "$order" "$SLURM_ARRAY_TASK_ID" "${bank[@]}"
