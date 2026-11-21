#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
stage=${1:?missing stage}
run=${SLURM_SUBMIT_DIR:?}
source "$WORK/ni26/.venv312/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$run/src/2026-competition"
cd "$run/src"
check=submissions/final_stack/check_cb_speed.py
data=data/proteus_windows.npz
if [[ $stage == smoke ]]; then
    for suffix in "" _r1a; do
        name=final_stack_hardcb_safe_st$suffix
        python -B "$check" smoke "$run/zips" "$data" "$run/smoke$suffix.json" --name "$name"
    done
elif [[ $stage == bench ]]; then
    python -B "$check" bench "$run/zips" "$data" "$run/bench-$SLURM_JOB_ID.json"
elif [[ $stage == options-smoke || $stage == options ]]; then
    flags=()
    if [[ $stage == options-smoke ]]; then flags+=(--smoke); fi
    python -B submissions/final_stack/cb_speed_options.py "$run/zips" "$data" \
        "$run/$stage-$SLURM_JOB_ID.json" "${flags[@]}"
elif [[ $stage == terminal ]]; then
    task=${SLURM_ARRAY_TASK_ID:?expected task 0-7}
    suffix="" order=recording
    if ((task / 4)); then suffix=_r1a; fi
    if ((task / 2 % 2)); then order=71000; fi
    python -B "$check" terminal "$run/zips" "$data" "$run/terminal-$SLURM_ARRAY_JOB_ID-$task.json" \
        --name "final_stack_hardcb_safe_st$suffix" --order "$order" --part "$((task % 2))" 2
else
    exit 2
fi
