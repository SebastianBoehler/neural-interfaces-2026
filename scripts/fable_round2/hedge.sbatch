#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:15:00
#SBATCH --job-name=fable2-p3
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
src="$run/hedge-code/scripts/fable_round2"
phase=${1:?}
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
case "$phase" in
  smoke|select|confirm)
    task=${SLURM_ARRAY_TASK_ID:-0}
    IFS=$'\t' read -r kind held _ < <(sed -n "$((task % 12 + 1))p" "$TB_SRC/hardcb/jobs.tsv")
    fold=${2:-${kind}_${held}}
    bash "$launcher" "$src/evaluate.py" "$phase" --run "$run" --data "$CAP250_DATA" \
      --fold "$fold" --chunk "$((task / 12))"
    ;;
  select-collect|confirm-collect)
    bash "$launcher" "$src/hedge_collect.py" "${phase%-collect}" --run "$run" --data "$CAP250_DATA"
    ;;
  *) exit 2 ;;
esac
