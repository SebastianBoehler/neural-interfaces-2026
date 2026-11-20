#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --job-name=laneTB-fold
set -euo pipefail
source "${TB_RUN:?}/env.sh"
if [[ ${1:-} == smoke ]]; then
  kind=smoke
  held=P01
  f=new_P01
  extra=(--stock "$TB_SRC/stock/$f/pkg"
         --safe "${TB_SRC%/*}/overnight-21nov-pkgcmp/hardcb_safe/$f/pkg")
else
  IFS=$'\t' read -r kind held _ < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TB_SRC/hardcb/jobs.tsv")
  f=${kind}_${held}
  extra=()
fi
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$TB_CODE/scripts/laneTB/run.py" \
  --source "$TB_SRC/hardcb/$f/pkg" --weights "$TB_RUN/tb/$f/weights.pt" --data "$CAP250_DATA" \
  --out "$TB_RUN/streams/${1:-$f}" --mode "$kind" --held "$held" "${extra[@]}"
