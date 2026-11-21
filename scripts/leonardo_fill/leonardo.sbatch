#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:20:00
#SBATCH --job-name=lf-replay
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
src="$run/src/scripts/leonardo_fill"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
phase=${1:?}
case "$phase" in
  dose-smoke)
    bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-${SLURM_JOB_ID}.log"
    grep -qx OK "$run/v3-${SLURM_JOB_ID}.log"
    bash "$launcher" "$src/dose.py" --fold "${2:?}" --data "$CAP250_DATA" --out "$run/dose"
    ;;
  dose)
    task=${SLURM_ARRAY_TASK_ID:?}
    IFS=$'\t' read -r kind held _ < <(sed -n "$((task % 12 + 1))p" "$TB_SRC/hardcb/jobs.tsv")
    bash "$launcher" "$src/dose.py" --fold "${kind}_${held}" --first "$((71400 + 2 * (task / 12)))" \
      --data "$CAP250_DATA" --out "$run/dose"
    ;;
  dose-report)
    half=()
    [[ -z ${2:-} ]] || half=(--half "$2")
    bash "$launcher" "$src/report.py" --data "$CAP250_DATA" --root "$run/dose" \
      --json "$run/dose${2:-}.json" "${half[@]}"
    ;;
  *)
    exit 2
    ;;
esac
