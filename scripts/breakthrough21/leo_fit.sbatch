#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:12:00
#SBATCH --job-name=bt21-fit
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
exec bash "$CAP250_RELEASE/cleanslate/cap250/slurm.sh" "$run/src/scripts/breakthrough21/fit.py" \
    "$CAP250_DATA" "${1:?fold}" "$run/fits/${1}.npz"
