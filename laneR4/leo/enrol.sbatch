#!/bin/bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 00:40:00
#SBATCH -J r4-enrol
# usage (R4 dir): sbatch --array=0-6 enrol.sbatch ; task -> variant; writes enrolled/V_k{0..8}.zip
set -euo pipefail
R4=/leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120; cd $R4
V=(hardcbB hardcbB_R1 hardcbB_R2 hardcbB_R3 hardcbB_R4 hardcbB_R5 hardcbB_R1a); v=${V[$SLURM_ARRAY_TASK_ID]}
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
export PYTHONPATH=$R4/stubs
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
mkdir -p enrolled logs
python -B niv_leo/build_enrolled.py zips/$v.zip enrolled $v
