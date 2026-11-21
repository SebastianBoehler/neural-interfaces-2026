#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:55:00
#SBATCH --job-name=fin-verify
# usage (run dir): sbatch --array=0-11 fin_verify.sbatch OUTTAG ORDERS ARMS [WEIGHTS_ROOT]   (array = jobs.tsv row, hard base)
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
IFS=$'\t' read -r kind held _ < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TB_SRC/hardcb/jobs.tsv")
w=()
[[ -z ${4:-} ]] || w=(--weights "$4")
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$run/code/scripts/laneTB/verify.py" --base hard \
  --fold "${kind}_${held}" --data "$CAP250_DATA" --out "$run/out/${1:?}" --orders "${2:?}" --arms "${3:?}" "${w[@]}"
