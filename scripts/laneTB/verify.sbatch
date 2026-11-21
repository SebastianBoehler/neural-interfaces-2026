#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:55:00
#SBATCH --job-name=laneTB-verify
# usage (from the verify run dir): sbatch [--array=0-11] verify.sbatch BASE ORDERS|smoke ; array task = jobs.tsv row
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
base=${1:?hardcb|hard}
if [[ ${2:?} == smoke ]]; then
  fold=new_P01; extra=(--smoke)
else
  IFS=$'\t' read -r kind held _ < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TB_SRC/hardcb/jobs.tsv")
  fold=${kind}_${held}; extra=(--orders "$2")
fi
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$run/code/scripts/laneTB/verify.py" \
  --base "$base" --fold "$fold" --data "$CAP250_DATA" --out "$run/out" "${extra[@]}"
