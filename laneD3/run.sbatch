#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 03:00:00
#SBATCH -J d3-run
# usage (release root): sbatch --array=0-18 laneD3/run.sbatch ; task = person. Orders/days/arms via B10_ORDERS / B10_DAYS / D3_ARMS (smoke overrides), D3_OUT (out dir).
set -euo pipefail
R=/leonardo_work/AIFAC_P02_1675/ni26_laneD3_20261120; cd $R
Z=/leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120/zips
k=${SLURM_ARRAY_TASK_ID}
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
export PYTHONPATH=$R/stubs OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
export B10_OFF=/leonardo_work/AIFAC_P02_1636/ni26/2026-competition/tracks/bci_decoding
export B10_ORDERS=${B10_ORDERS:-full:72000,full:72001}
d=${D3_OUT:-out}/k$k; mkdir -p $d logs
[ -e $d/d3.npz ] && { echo exists $d; exit 0; }
python -B laneD3/run.py $Z/hardcbB.zip ext $k $d/d3.npz
