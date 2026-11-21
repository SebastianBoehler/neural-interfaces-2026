#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --job-name=bt21-score
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
src="$run/src/scripts/breakthrough21"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
stage=${1:?}
case "$stage" in
    smoke)
        bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/logs/v3-$SLURM_JOB_ID.txt"
        grep -qx OK "$run/logs/v3-$SLURM_JOB_ID.txt"
        exec bash "$launcher" "$src/worker.py" "$stage" "$run" "$CAP250_DATA" "${2:?fold}"
        ;;
    screen|held|fit)
        i=${SLURM_ARRAY_TASK_ID:?}
        IFS=$'\t' read -r mode person _ < <(sed -n "$((i % 12 + 1))p" "$TB_SRC/hardcb/jobs.tsv")
        fold="${mode}_${person}"
        if [[ "$stage" == fit ]]; then
            [[ "$fold" != new_P01 ]] || exit 0
            exec bash "$launcher" "$src/fit.py" "$CAP250_DATA" "$fold" "$run/fits/$fold.npz"
        fi
        exec bash "$launcher" "$src/worker.py" "$stage" "$run" "$CAP250_DATA" "$fold" "$((i / 12))"
        ;;
    collect)
        exec bash "$launcher" "$src/collect.py" "${2:?block}" "$run" "$CAP250_DATA"
        ;;
    *) exit 2 ;;
esac
