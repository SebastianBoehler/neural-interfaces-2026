#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
history="$WORK/ni26/cap250/runs"
base="$history/leonardo-fill-ba-six-21nov"
source "$base/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
mode=${1:?}
task=()
if [[ $mode == smoke ]]; then
    bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
    grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
    task=(--task "${SLURM_ARRAY_TASK_ID:?}")
elif [[ $mode != report ]]; then
    task=(--task "${SLURM_ARRAY_TASK_ID:?}")
fi
exec bash "$launcher" "$run/src/scripts/bottleneck_21nov/arm_c_sprint_proteus.py" \
    "$mode" "$CAP250_DATA" "$base/grow" "$history" "$run/out" "${task[@]}"
