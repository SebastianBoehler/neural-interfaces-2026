#!/bin/bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 02:00:00
#SBATCH -J r4-niv
# usage (R4 dir): sbatch --array=0-62 run.sbatch ; task = variant*9 + person; Scherer B-ret on NEW orders (explicit seeds 71008-71023, full and within, visit 2)
set -euo pipefail
R4=/leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120; cd $R4
V=(hardcbB hardcbB_R1 hardcbB_R2 hardcbB_R3 hardcbB_R4 hardcbB_R5 hardcbB_R1a); t=$SLURM_ARRAY_TASK_ID; v=${V[$((t/9))]}; k=$((t%9))
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
export PYTHONPATH=$R4/stubs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NIV_VISITS=2
arms=$(python3 -c "print(','.join(f'{s}:s{71008+i}' for s in ('full','within') for i in range(16)))")
mkdir -p out/${v}_k$k logs
out=out/${v}_k$k/ret.npz; [ -e $out ] && { echo exists $out; exit 0; }
python -B niv_leo/worker.py enrolled/${v}_k$k.zip $out $k "$arms"
