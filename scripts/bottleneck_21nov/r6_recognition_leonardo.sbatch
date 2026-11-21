#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:08:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
all="$WORK/ni26/cap250/runs"
base="$all/leonardo-fill-ba-six-21nov"
source "$base/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
worker="$run/src/scripts/bottleneck_21nov/r6_recognition_capture.py"
common=("$CAP250_DATA" "$base/grow" "$all/bottleneck-returning-em-21nov/out" "$run/out")
if [[ ${1:?} == smoke ]]; then
    bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
    grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
    exec bash "$launcher" "$worker" "${common[@]}" ret_P04 71800 --smoke
fi
[[ $1 == capture ]]
i=${SLURM_ARRAY_TASK_ID:?}
folds=(ret_P04 ret_P10)
exec bash "$launcher" "$worker" "${common[@]}" "${folds[i % 2]}" "$((71800 + 4 * (i / 2)))"
