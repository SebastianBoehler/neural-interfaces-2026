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
all="$WORK/ni26/cap250/runs"
base="$all/leonardo-fill-ba-six-21nov"
source "$base/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
hss="$all/final-r1a-21nov/zips/final_stack_hard_safe_st.zip"
options=()
if [[ ${1:?} == smoke ]]; then
    bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
    grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
    options=(--smoke --hss-pin "$all/laneTB-verify-21nov/out/hard")
fi
exec bash "$launcher" "$run/src/scripts/fable_r8/audit.py" "$CAP250_DATA" "$base/grow" "$hss" "$run/out" \
    "${SLURM_ARRAY_TASK_ID:?}" "${options[@]}"
