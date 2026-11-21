#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=00:20:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
src="$run/src-size/scripts/breakthrough21"
export PYTHONPATH="$run/src-size/scripts/laneTB${PYTHONPATH:+:$PYTHONPATH}"
stage=${1:?}
case "$stage" in
  smoke) fold=new_P01 ;;
  capture)
    IFS=$'\t' read -r mode person _ < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TB_SRC/hardcb/jobs.tsv")
    fold="${mode}_${person}"
    ;;
  report) fold="" ;;
  *) exit 2 ;;
esac
exec bash "$launcher" "$src/size.py" "$stage" "$run" "$CAP250_DATA" "$fold"
