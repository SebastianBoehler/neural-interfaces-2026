#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:25:00
#SBATCH --job-name=lf-grow
set -euo pipefail
root=${SLURM_SUBMIT_DIR:?}
run="$root/grow"
source "$root/env.sh"
src="$root/src-grow/scripts/leonardo_fill"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
phase=${1:?}
wave=${LF_GROW_WAVE:-growth}
case "$phase" in
  smoke)
    bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-${SLURM_JOB_ID}.log"
    grep -qx OK "$run/v3-${SLURM_JOB_ID}.log"
    fold=${2:?}
    ;;
  prepare|screen|held)
    task=${SLURM_ARRAY_TASK_ID:?}
    IFS=$'\t' read -r kind held _ < <(sed -n "$((task % 12 + 1))p" "$TB_SRC/hardcb/jobs.tsv")
    fold="${kind}_${held}"
    ;;
  timing)
    task=${SLURM_ARRAY_TASK_ID:?}
    folds=(new_P01 ret_P04)
    arms=(swap25 fbc12 ba_six)
    if [[ $wave == roster ]]; then arms=(triseed25 ba_seven ba_eight); fi
    fold=${folds[task / 3]}
    arm=${arms[task % 3]}
    ;;
  select|report)
    if [[ $phase == report ]]; then phase=held; fi
    exec bash "$launcher" "$src/grow_report.py" "$phase" --run "$run" --data "$CAP250_DATA" --wave "$wave"
    ;;
  *)
    exit 2
    ;;
esac
args=(--chunk "$((${SLURM_ARRAY_TASK_ID:-0} / 12))")
if [[ $phase == timing ]]; then args+=(--arm "$arm"); fi
exec bash "$launcher" "$src/grow_replay.py" "$phase" --fold "$fold" --run "$run" --data "$CAP250_DATA" --wave "$wave" "${args[@]}"
