#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
phase=${1:?}
if [[ $phase == folds ]]; then
    folds=(new_P02 new_P03 new_P04 new_P05 new_P06 new_P07 new_P08 new_P09 new_P10 ret_P04 ret_P10 production niv-production)
    phase=${folds[${SLURM_ARRAY_TASK_ID:?}]}
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$run/src/2026-competition"
"$WORK/ni26/.venv312/bin/python" -B "$run/src/submissions/final_stack/audit_ba_terminal.py" \
    /leonardo_work/AIFAC_P02_1675/ni26_cleanslate_A2 "$run/src/data/proteus_windows.npz" \
    "$WORK/ni26/cap250/runs/laneTB-grow-21nov" "$phase" "$run/out/audit_$phase.json"
