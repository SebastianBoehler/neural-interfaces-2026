#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:18:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
all="$WORK/ni26/cap250/runs"
base="$all/leonardo-fill-ba-six-21nov"
source "$base/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
worker="$run/src/scripts/bottleneck_21nov/final_review_capture.py"
common=("$CAP250_DATA" "$base/grow" "$all/bottleneck-returning-em-21nov/out" "$run/em_archive_manifest.json")
case "${1:?}" in
    smoke)
        bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
        grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
        exec bash "$launcher" "$worker" "${common[@]}" "$run/smoke" ret_P04 fit --smoke
        ;;
    capture)
        folds=(ret_P04 ret_P10 new_P10 ret_P04 ret_P10)
        phases=(fit fit fit held held)
        i=${SLURM_ARRAY_TASK_ID:?}
        exec bash "$launcher" "$worker" "${common[@]}" "$run/out" "${folds[i]}" "${phases[i]}"
        ;;
    *) exit 2 ;;
esac
