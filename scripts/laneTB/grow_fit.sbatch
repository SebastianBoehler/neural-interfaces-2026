#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --job-name=tb-grow-fit
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
IFS=$'\t' read -r kind held people first _ < <(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$TB_SRC/hardcb/jobs.tsv")
fold=${kind}_${held}
launch=$CAP250_RELEASE/cleanslate/cap250/slurm.sh
code=$run/code/cleanslate
case ${1:?} in
  b-smoke|b)
    unset CSB_EXT CSB_EXPERT_STEPS
    export CSB_STEPS=1200 CSB_NETS=shallow:3,slow:3,fbc:3 CSB_LAM=0.5 CSB_LAM_EXT=1
    target=$run/b/$fold
    if [[ $1 == b-smoke ]]; then
      export CSB_STEPS=30 CSB_NETS=shallow:1,slow:1,fbc:1
      target=$run/fit-smoke/b
    fi
    [[ $kind == ret ]] || first=
    test ! -e "$target/weights.pt"
    bash "$launch" "$code/teamB/train.py" --people "$people" --first-session-only "$first" \
      --out "$target" --data "$CAP250_DATA" --seed 1
    sha256sum "$target/weights.pt" "$target/submission.py"
    ;;
  c-smoke|c)
    [[ $kind == ret && $first == "$held" ]]
    epochs=60
    target=$run/c/$fold
    if [[ $1 == c-smoke ]]; then
      epochs=2
      target=$run/fit-smoke/c
    fi
    test ! -e "$target/weights.pt"
    bash "$launch" "$code/teamC/train.py" --people "$people" --exclude "$held|2,$held|3" \
      --out "$target" --epochs "$epochs" --data "$CAP250_DATA"
    sha256sum "$target/weights.pt" "$target/train.json" "$target/submission.py"
    ;;
  *) exit 2 ;;
esac
