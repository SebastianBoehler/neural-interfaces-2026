#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:16:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
all="$WORK/ni26/cap250/runs"
base="$all/leonardo-fill-ba-six-21nov"
prior="$all/fable-r6-diet-21nov"
source "$base/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
src="$run/src/scripts/bottleneck_21nov"
common=("$CAP250_DATA" "$base/grow")
case "${1:?}" in
    smoke)
        bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
        grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
        bash "$launcher" "$src/returning_diet_worker.py" "${common[@]}" \
            "$all/bottleneck-returning-em-21nov/out" "$prior/em_archive_manifest.json" "$run/smoke" 22 --smoke
        exec bash "$launcher" "$src/quarter_fresh.py" "${common[@]}" "$run/smoke" 10 --smoke
        ;;
    score)
        exec bash "$launcher" "$src/quarter_fresh.py" "${common[@]}" "$run/out" "${SLURM_ARRAY_TASK_ID:?}"
        ;;
    report)
        exec bash "$launcher" "$src/quarter_fresh_report.py" "${common[@]}" "$run/out" "$run/report.json"
        ;;
    *) exit 2 ;;
esac
