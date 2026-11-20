#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=laneTB-fit
set -euo pipefail
source "$TB_RUN/env.sh"
if [[ ${1:-} == smoke ]]; then
  fold=new_P01
  people=P02,P03,P04,P05,P06,P07,P08,P09,P10
  first=
  out=$TB_RUN/smoke/tb
  export CSB_STEPS=30 CSB_NETS=shallow:1,slow:1,fbc:1
else
  IFS=$'\t' read -r kind held people first sessions < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TB_SRC/hardcb/jobs.tsv")
  fold=${kind}_${held}
  out=$TB_RUN/tb/$fold
  [[ $kind == ret ]] || first=
  export CSB_STEPS=1200 CSB_NETS=shallow:3,slow:3,fbc:3
fi
unset CSB_EXT CSB_EXPERT_STEPS
export CSB_LAM=0.5 CSB_LAM_EXT=1
mkdir -p "$out"
test ! -e "$out/weights.pt"
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$TB_CODE/cleanslate/teamB/train.py" \
  --people "$people" --first-session-only "$first" --out "$out" --data "$CAP250_DATA" --seed 0
sha256sum "$out/weights.pt" "$out/submission.py" | tee "$out/sha256"
