#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
all="$WORK/ni26/cap250/runs"
base="$all/leonardo-fill-ba-six-21nov"
source "$base/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export PYTHONPATH="$all/bottleneck-21nov-r2/src/scripts/bottleneck_21nov${PYTHONPATH:+:$PYTHONPATH}"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
worker="$run/src/new_cohort_capture.py"
out="$run/out"
case "${1:?}" in
    smoke)
        bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
        grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
        exec bash "$launcher" "$worker" "$CAP250_DATA" "$base/grow" "$out" new_P01 71804 --smoke
        ;;
    capture)
        i=${SLURM_ARRAY_TASK_ID:?}
        folds=(new_P01 new_P02 new_P03 new_P04 new_P05 new_P06 new_P07 new_P08 new_P09 new_P10 ret_P04 ret_P10)
        starts=(71804 71806 71812 71814)
        exec bash "$launcher" "$worker" "$CAP250_DATA" "$base/grow" "$out" "${folds[i % 12]}" "${starts[i / 12]}"
        ;;
    count-control)
        worker="$run/control/new_cohort_capture.py"
        tail="$all/fable-r6-diet-21nov/src/scripts/last_push/tape.py"
        exec bash "$launcher" "$worker" "$CAP250_DATA" "$base/grow" "$out" new_P01 71800 --count-tail "$tail"
        ;;
    *) exit 2 ;;
esac
