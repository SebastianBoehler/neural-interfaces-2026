#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=laneTB-collect
set -euo pipefail
source "${TB_RUN:?}/env.sh"
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$TB_CODE/scripts/laneTB/collect.py" \
  "$TB_RUN/streams" --data "$CAP250_DATA" --out "$TB_RUN/result.json"
