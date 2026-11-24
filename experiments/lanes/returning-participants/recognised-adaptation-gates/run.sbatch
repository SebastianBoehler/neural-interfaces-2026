#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 03:00:00
#SBATCH -J r4b-run
# usage (release root): SET=net sbatch --array=0-N laneR4b/run.sbatch ; task = variant*P + person (net: P 19, ds7: P 14). VARS (space list) and ORD overrides for gated waves.
set -euo pipefail
R=/leonardo_work/AIFAC_P02_1675/ni26_laneR4b_20261120; cd $R
SET=${SET:?}; if [ $SET = net ]; then P=19; EXT=$R/ext; PAIRS="1>2,3,4"; ORD=${ORD:-full:72000,full:72001,within:72000,within:72001}; else P=14; EXT=$R/ext7; PAIRS="1>2;2>1"; ORD=${ORD:-full:72000,full:72001,full:72002,full:72003,within:72000,within:72001,within:72002,within:72003}; fi
read -ra V <<< "${VARS:?}"
t=${SLURM_ARRAY_TASK_ID}; v=${V[$((t/P))]}; k=$((t%P))
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
export PYTHONPATH=$R/stubs OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
export B10_OFF=/leonardo_work/AIFAC_P02_1636/ni26/2026-competition/tracks/bci_decoding B10_ORDERS=$ORD R4B_PAIRS=$PAIRS
d=${R4B_OUT:-out}/${SET}_${v}_k$k; mkdir -p $d logs
[ -e $d/ret.npz ] && { echo exists $d; exit 0; }
python -B laneR4b/run.py $R/zips/$v.zip $EXT $k $d/ret.npz
