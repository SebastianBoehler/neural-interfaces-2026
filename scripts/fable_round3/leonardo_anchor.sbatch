#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=fable3-anchor
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
base=$(dirname "$run")
source "$base/env.sh"
src="$run/src/scripts/fable_round3"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
stage=${1:?}
task=${SLURM_ARRAY_TASK_ID:-0}
case "$stage" in
  smoke) fold=${2:?}; chunk=0 ;;
  bank|held)
    IFS=$'\t' read -r kind person _ < <(sed -n "$((task % 12 + 1))p" "$TB_SRC/hardcb/jobs.tsv")
    fold=${kind}_${person}
    chunk=$((task / 12))
    ;;
  fit)
    people=(ret_P04 ret_P10)
    fold=${people[task % 2]}
    chunk=$((task / 2))
    ;;
  select|confirm)
    exec bash "$launcher" "$src/anchor_collect.py" "$stage" --base "$base" --run "$run" --data "$CAP250_DATA"
    ;;
  *) exit 2 ;;
esac
bash "$launcher" "$src/anchor_worker.py" "$stage" --base "$base" --run "$run" --data "$CAP250_DATA" \
    --fold "$fold" --chunk "$chunk"
