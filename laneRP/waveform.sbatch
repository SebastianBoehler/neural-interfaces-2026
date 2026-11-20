#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=logs/waveform-%j.out
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
root=/leonardo_work/AIFAC_P02_1636/ni26
net=/leonardo_work/AIFAC_P02_1675/ni26_laneB10_20261120
zip=/leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120/zips/v66_cp1_cap250.zip
args=(--features features --pred out --out wave --identity bio_model_psd_d64.npz --zip "$zip" --stock "$root/cap250/runs/production-20261120-32a94ab7/stock" --public "$root/data/proteus_windows.npz" --scherer "$root/data/scherer2015/windows.npz" --b10 "$net")
module load python/3.11.7
"$root/.venv312/bin/python" laneRP/wave_replay.py "${args[@]}" --smoke
exec "$root/.venv312/bin/python" laneRP/wave_replay.py "${args[@]}"
