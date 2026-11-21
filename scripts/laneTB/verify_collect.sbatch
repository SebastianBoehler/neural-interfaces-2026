#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=laneTB-vcollect
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$run/code/scripts/laneTB/verify_collect.py" "$run/out" \
  --data "$CAP250_DATA" --orders "${1:?orders}" --tb "$TB_RUN/streams" --json "$run/result.json"
