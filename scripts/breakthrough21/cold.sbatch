#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
code="$run/src-cold"
src="$code/scripts/breakthrough21"
export PYTHONPATH="$code/scripts/laneTB${PYTHONPATH:+:$PYTHONPATH}"
stage=${1:?}
case "$stage" in
  fit) fold=${2:?} ;;
  fit-folds|cold_screen|cold_held)
    i=${SLURM_ARRAY_TASK_ID:?}
    IFS=$'\t' read -r mode person _ < <(sed -n "$((i % 12 + 1))p" "$TB_SRC/hardcb/jobs.tsv")
    fold="${mode}_${person}"
    ;;
  smoke|report) fold=${2:?} ;;
  *) exit 2 ;;
esac
if [[ "$stage" == fit || "$stage" == fit-folds ]]; then
    [[ "$stage" != fit-folds || "$fold" != new_P01 ]] || exit 0
    exec bash "$launcher" "$src/cold_prior.py" "$CAP250_DATA" "$fold" "$run/priors/$fold.npy"
fi
exec bash "$launcher" "$src/cold_worker.py" "$stage" "$run" "$CAP250_DATA" "$fold" "$(("${SLURM_ARRAY_TASK_ID:-0}" / 12))"
