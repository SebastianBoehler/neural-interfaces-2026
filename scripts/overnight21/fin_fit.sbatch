#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=fin-fit
# usage (from the run dir, array = jobs.tsv row 0-11): sbatch --array=0-11 fin_fit.sbatch SEED [smoke]
# Team B nine (3 shallow, 3 slow, 3 fbc; 1200 steps) on production-20261120-32a94ab7's fold rosters, same recipe as the
# seed-0 fits and the seed-1 TB-grow fits (grow_fit.sbatch b), only --seed differs. Output $run/b_s<SEED>/<fold>/weights.pt
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
seed=${1:?}
IFS=$'\t' read -r kind held people first _ < <(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$TB_SRC/hardcb/jobs.tsv")
fold=${kind}_${held}
[[ $kind == ret ]] || first=
unset CSB_EXT CSB_EXPERT_STEPS
export CSB_STEPS=1200 CSB_NETS=shallow:3,slow:3,fbc:3 CSB_LAM=0.5 CSB_LAM_EXT=1
target=$run/b_s$seed/$fold
if [[ ${2:-} == smoke ]]; then
  export CSB_STEPS=30 CSB_NETS=shallow:1,slow:1,fbc:1
  target=$run/fit-smoke/s$seed
fi
test ! -e "$target/weights.pt"
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$run/code/cleanslate/teamB/train.py" --people "$people" \
  --first-session-only "$first" --out "$target" --data "$CAP250_DATA" --seed "$seed"
sha256sum "$target/weights.pt"
