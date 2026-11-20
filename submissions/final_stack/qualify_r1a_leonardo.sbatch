#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
stage=${1:?smoke or pair}
mode=${2:?hard or hardcb}
run=${SLURM_SUBMIT_DIR:?}
work="$WORK/ni26"
source "$work/.venv312/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NI26_MAX_MB=100
export PYTHONPATH="$run/src/2026-competition"
cd "$run/src"
zips=$run/zips
if [[ $stage == smoke ]]; then
  python -B submissions/final_stack/check_r1a.py "$mode" "$zips" data/proteus_windows.npz \
    "$run/smoke-$mode.json" --device cuda:0 --smoke
elif [[ $stage == heldnew ]]; then
  python -B submissions/final_stack/check_r1a.py "$mode" "$zips" data/proteus_windows.npz \
    "$run/heldnew-$mode-$SLURM_JOB_ID.json" --device cpu \
    --fold "$work/cap250/runs/production-20261120-32a94ab7/hardcb/new_P01/pkg"
  python -B submissions/final_stack/selftest_inject.py "$zips/final_stack_${mode}_safe_st_r1a.zip" "$mode"
elif [[ $stage == pair ]]; then
  cand="$zips/final_stack_${mode}_safe_st_r1a.zip"
  python -B friday/qualify_package.py pair "$cand" "$run/qual-$mode-$SLURM_JOB_ID" \
    --ref-zip "$WORK/ni26_laneR4_20261120/zips/v66_cp1_cap250.zip" --expect-sha "${3:?candidate SHA}"
elif [[ $stage == returning ]]; then
  python -B submissions/final_stack/check_r1a.py "$mode" "$zips" data/proteus_windows.npz \
    "$run/return-$mode-$SLURM_JOB_ID.json" --device cuda:0 \
    --tapes "$work/cap250/runs/redteam-21nov/R1a"
else
  exit 2
fi
