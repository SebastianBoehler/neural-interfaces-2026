#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/expert-%j.out
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
root=/leonardo_work/AIFAC_P02_1636/ni26
net=/leonardo_work/AIFAC_P02_1675/ni26_laneB10_20261120
args=(--features features --out out --population out --stock "$root/cap250/runs/production-20261120-32a94ab7/stock" --public "$root/data/proteus_windows.npz" --scherer "$root/data/scherer2015/windows.npz" --manifest scherer_rows.tsv.gz --niv niv --b10 "$net")
module load python/3.11.7
"$root/.venv312/bin/python" laneRP/experiment.py "$1" "${args[@]}" --smoke
exec "$root/.venv312/bin/python" laneRP/experiment.py "$1" "${args[@]}"
