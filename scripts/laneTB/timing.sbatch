#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --job-name=laneTB-timing
set -euo pipefail
source "${TB_RUN:?}/env.sh"
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$TB_CODE/scripts/laneTB/timing.py" \
  --source "$TB_SRC" --run "$TB_RUN" --data "$CAP250_DATA"
