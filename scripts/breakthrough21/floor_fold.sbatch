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
src="$run/src-floor/scripts/breakthrough21"
stage=${1:?}
i=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$stage" == score ]]; then
    IFS=$'\t' read -r mode person _ < <(sed -n "$((i % 12 + 1))p" "$TB_SRC/hardcb/jobs.tsv")
    fold="${mode}_${person}"
else
    fold=${2:-all}
fi
exec bash "$launcher" "$src/floor_fold.py" "$stage" "$run" "$CAP250_DATA" "$fold" "$((i / 12))"
