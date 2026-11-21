#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:25:00
#SBATCH --job-name=fable2
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
src="$run/code/scripts/fable_round2"
run_py() {
  bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$src/$1" "${@:2}"
}
case "${1:?smoke|ocular|encoder|collect}" in
  smoke)
    NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data" run_py "$WORK/ni26/replay/check.py" | tee "$run/v3.log"
    grep -qx OK "$run/v3.log"
    run_py diagnose.py --data "$CAP250_DATA" --smoke
    run_py live.py --data "$CAP250_DATA" --out "$run/out" --smoke
    ;;
  ocular)
    run_py diagnose.py --data "$CAP250_DATA" --out "$run/ocular.json" --score
    ;;
  encoder)
    IFS=$'\t' read -r kind held _ < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TB_SRC/hardcb/jobs.tsv")
    run_py live.py --data "$CAP250_DATA" --out "$run/out" --fold "${kind}_${held}"
    ;;
  collect)
    run_py diagnose.py --data "$CAP250_DATA" --out "$run/diagnostics.json" --score --encoder "$run/out"
    ;;
  *)
    exit 2
    ;;
esac
