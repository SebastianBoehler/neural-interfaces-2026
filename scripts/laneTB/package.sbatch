#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:40:00
#SBATCH --job-name=laneTB-package
set -euo pipefail
source "${TB_RUN:?}/env.sh"
if ! grep -Eq '"package_arm": "(nine|three)(15|25)"' "$TB_RUN/result.json"; then
  grep -F '"package_arm": null' "$TB_RUN/result.json"
  echo "No positive held-new blend; ZIP build skipped."
  exit 0
fi
export CSB_STEPS=1200 CSB_NETS=shallow:3,slow:3,fbc:3 CSB_LAM=0.5 CSB_LAM_EXT=1
unset CSB_EXT CSB_EXPERT_STEPS
people=P01,P02,P03,P04,P05,P06,P07,P08,P09,P10
out=$TB_RUN/package
mkdir -p "$out"
test ! -e "$out/weights.pt"
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$TB_CODE/cleanslate/teamB/train.py" \
  --people "$people" --out "$out" --data "$CAP250_DATA" --seed 0
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$TB_CODE/scripts/laneTB/package.py" \
  --base "$TB_RUN/final_stack_hardcb_safe.zip" --bank "$out/weights.pt" \
  --result "$TB_RUN/result.json" --data "$CAP250_DATA" --outdir "$out"
