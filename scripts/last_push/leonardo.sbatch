#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:35:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
base="$WORK/ni26/cap250/runs/leonardo-fill-ba-six-21nov"
source "$base/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
script="$run/src/scripts/last_push"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
phase=${1:?}
case "$phase" in
    smoke)
        bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
        grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
        fold=${2:?}
        roster=${3:?}
        chunk=0
        ;;
    fit|held)
        i=${SLURM_ARRAY_TASK_ID:?}
        IFS=$'\t' read -r mode person _ < <(sed -n "$((i % 12 + 1))p" "$TB_SRC/hardcb/jobs.tsv")
        fold="${mode}_${person}"
        rosters=(nine25 ba_six)
        roster=${rosters[i / 12 % 2]}
        chunk=$((i / 24))
        ;;
    report)
        exec bash "$launcher" "$script/report.py" "${2:?}" "$run" "$CAP250_DATA"
        ;;
    *) exit 2 ;;
esac
exec bash "$launcher" "$script/replay.py" "$phase" "$CAP250_DATA" "$roster" \
    --fixture "$base/grow" --run "$run" --fold "$fold" --chunk "$chunk"
