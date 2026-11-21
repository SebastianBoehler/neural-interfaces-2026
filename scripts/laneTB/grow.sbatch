#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:20:00
#SBATCH --job-name=tb-grow
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
phase=${1:?}
job=${SLURM_ARRAY_TASK_ID:-0}
code=$run/code/scripts/laneTB
launcher=$CAP250_RELEASE/cleanslate/cap250/slurm.sh
case "$phase" in
  select-collect|confirm-collect|extra-collect)
    bash "$launcher" "$code/grow_collect.py" "${phase%-collect}" --run "$run" --data "$CAP250_DATA"
    ;;
  smoke|returning-smoke|timing|select|select-repair|confirm|extra|finalists)
    extra=()
    if [[ $phase == select-repair ]]; then
      phase=select
      extra=(--resume)
    fi
    IFS=$'\t' read -r kind who _ < <(sed -n "$(( job % 12 + 1 ))p" "$TB_SRC/hardcb/jobs.tsv")
    bash "$launcher" "$code/grow_run.py" "$phase" --fold "${kind}_${who}" --chunk "$(( job / 12 ))" \
      --run "$run" --data "$CAP250_DATA" "${extra[@]}"
    ;;
  *) exit 2 ;;
esac
