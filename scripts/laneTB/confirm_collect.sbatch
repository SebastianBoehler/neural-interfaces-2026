#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --job-name=laneTB-cconfirm
# from laneTB-confirm-21nov: sbatch [--dependency=afterany:IDS] confirm_collect.sbatch
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$run/code/scripts/laneTB/confirm_collect.py" "$run/out" \
  "$(dirname "$run")/laneTB-verify-21nov/out" --data "$CAP250_DATA" \
  --orders 71156,71157,71158,71159,71160,71161,71162,71163,71164,71165,71166,71167,71168,71169,71170,71171 \
  --verify-orders 71124,71125,71126,71127,71128,71129,71130,71131 --json "$run/result.json"
