#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --job-name=laneTB-pkgcheck
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$run/code/scripts/laneTB/pkgcheck.py" \
  --zip "$TB_RUN/package/final_stack_hardcb_safe_team_b_nine25.zip" --base "$TB_RUN/final_stack_hardcb_safe.zip" \
  --data "$CAP250_DATA" --out "$run/pkgcheck.json"
