#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" \
    "$run/refill21_leonardo_report.py" "$run" --data "$CAP250_DATA"
