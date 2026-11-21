#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=fable-r4
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
export CAP250_RUN="$run"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
src="$run/src/scripts/fable_r4"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
stage=${1:?}
case "$stage" in
  smoke)
    fold=${2:?}
    chunk=0
    bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-${SLURM_JOB_ID}.log"
    grep -qx OK "$run/v3-${SLURM_JOB_ID}.log"
    ;;
  oracle)
    task=${SLURM_ARRAY_TASK_ID:?}
    fold=$(printf 'new_P%02d' "$((task % 10 + 1))")
    chunk=$((task / 10))
    ;;
  collect)
    exec bash "$launcher" "$src/score.py" "${2:?}" --run "$run" --data "$CAP250_DATA"
    ;;
  *) exit 2 ;;
esac
exec bash "$launcher" "$src/worker.py" "$stage" --run "$run" --data "$CAP250_DATA" --fold "$fold" --chunk "$chunk"
