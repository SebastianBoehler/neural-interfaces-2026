#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
team_a=/leonardo_work/AIFAC_P02_1675/ni26_cleanslate_A2
py="$WORK/ni26/.venv312/bin/python"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$run/src"
parent="$run/zips/final_stack_hard_safe_st_team_b_nine25_tbgraph.zip"
case ${1:?} in
    production)
        "$py" -B submissions/final_stack/build_ba_terminal.py "$parent" \
            --production "$team_a/b6-a1-manas/teamA-manas.zip" \
            "$run/zips/final_stack_hard_safe_st_team_b_nine25_ba_terminal.zip"
        ;;
    niv)
        "$py" -B submissions/final_stack/build_ba_terminal.py "$parent" \
            --scherer-free "$team_a/b6-a1-niv-manas/production" --scherer-free-sha "${2:?}" \
            "$run/zips/ba_terminal_niv.zip"
        ;;
    *) exit 2 ;;
esac
