#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-laneE-leo-time
#SBATCH --output=logs/leo-time-%j.out
# sbatch --account=ACC leo_timing.sbatch   (paired warm A100 timing of cap250, E0, Ea, Eb, Ec, Ed, Ed_g2; lane_e/timing.py)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset PYTHONPATH; module load python/3.11.7 >/dev/null 2>&1 || true
PY=${LEO_REPO:-/leonardo_work/AIFAC_P02_1636/ni26}/.venv/bin/python
(sha256sum --quiet -c zips.sha256 --ignore-missing)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' '); mkdir -p out
$PY -I -B code/lane_e/timing.py out/timing_${gpu}.json cuda:0 zips/v66_cp1_cap250.zip:cap250 zips/E0.zip:E0 zips/Ea.zip:Ea zips/Eb.zip:Eb \
  zips/Ec.zip:Ec zips/Ed.zip:Ed zips/Ed_g2.zip:Ed_g2
