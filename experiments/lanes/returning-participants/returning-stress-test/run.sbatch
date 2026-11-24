#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 03:00:00
#SBATCH -J b10-run
# usage (release root): sbatch --array=0-189 laneB10/run.sbatch ; task = variant*19 + person. Orders and days via B10_ORDERS / B10_DAYS (smoke overrides).
set -euo pipefail
R=/leonardo_work/AIFAC_P02_1675/ni26_laneB10_20261120; cd $R
Z=/leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120/zips
V=(v66_cp1_cap250 hardcbB hardcbB_R1 hardcbB_R2 hardcbB_R5 hardcbB_R1a hardcbB_R3 hardcbB_R4 v66_cp1_cap250:new hardcbB:new)
t=${SLURM_ARRAY_TASK_ID}; v=${V[$((t/19))]}; k=$((t%19)); base=${v%%:*}; flag=""; tag=$base; [ "$v" != "$base" ] && { flag="--new"; tag=${base}_new; }
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
export PYTHONPATH=$R/stubs OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
export B10_OFF=/leonardo_work/AIFAC_P02_1636/ni26/2026-competition/tracks/bci_decoding
export B10_ORDERS=${B10_ORDERS:-full:72000,full:72001,full:72002,full:72003,within:72000,within:72001,within:72002,within:72003}
d=${B10_OUT:-out}/${tag}_k$k; mkdir -p $d logs
[ -e $d/ret.npz ] && { echo exists $d; exit 0; }
python -B laneB10/run.py $Z/$base.zip ext $k $d/ret.npz $flag
