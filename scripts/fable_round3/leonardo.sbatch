#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:55:00
#SBATCH --job-name=fable3
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
src="$run/src/scripts/fable_round3"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
stage=${1:?}
case "$stage" in
  fold-smoke)
    bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-${SLURM_JOB_ID}.log"
    grep -qx OK "$run/v3-${SLURM_JOB_ID}.log"
    bash "$launcher" "$src/worker.py" "$stage" --data "$CAP250_DATA" --fold "${2:?}"
    ;;
  trace|grid|held)
    task=${SLURM_ARRAY_TASK_ID:?}
    IFS=$'\t' read -r kind held _ < <(sed -n "$((task % 12 + 1))p" "$TB_SRC/hardcb/jobs.tsv")
    bash "$launcher" "$src/worker.py" "$stage" --run "$run" --data "$CAP250_DATA" \
      --fold "${kind}_${held}" --chunk "$((task / 12))" --block "${2:?}"
    ;;
  screen|select|confirm)
    bash "$launcher" "$src/decisions.py" "$stage" --run "$run" --data "$CAP250_DATA"
    ;;
  *)
    exit 2
    ;;
esac
