#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
base="$WORK/ni26/cap250/runs/leonardo-fill-ba-six-21nov"
source "$base/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
worker="$run/src/scripts/bottleneck_21nov/capture.py"
folds=(new_P01 new_P02 new_P03 new_P04 new_P05 new_P06 new_P07 new_P08 new_P09 new_P10 ret_P04 ret_P10)
rosters=(nine25 ba_six)
case "${1:?}" in
    smoke)
        bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
        grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
        exec bash "$launcher" "$worker" "$CAP250_DATA" "$base/grow" "$run/smoke" "${2:?}" "${3:?}" 71800 --smoke
        ;;
    capture)
        i=${SLURM_ARRAY_TASK_ID:?}
        roster=${rosters[i / 12 % 2]}
        fold=${folds[i % 12]}
        chunks=(71800 71802 71808 71810)
        first=${chunks[i / 24]}
        exec bash "$launcher" "$worker" "$CAP250_DATA" "$base/grow" "$run/out" "$roster" "$fold" "$first" "$((first + 1))"
        ;;
    *) exit 2 ;;
esac
