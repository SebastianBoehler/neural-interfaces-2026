#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=logs/pop-%j.out
set -eu
root=/leonardo_work/AIFAC_P02_1636/ni26
export CAP250_RELEASE=$root/cap250/releases/15a639718601e959015fd533fa9106476614aa8c
export CS_OFFICIAL=$CAP250_RELEASE/2026-competition/tracks/bci_decoding
export PYTHONPATH=$CS_OFFICIAL:$CAP250_RELEASE/2026-competition
export MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
module load python/3.11.7
exec "$root/.venv312/bin/python" -B laneRP/population.py --stock "$root/cap250/runs/production-20261120-32a94ab7/stock" --data "$root/data/proteus_windows.npz" --out out "$@"
