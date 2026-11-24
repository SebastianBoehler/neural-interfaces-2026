#!/bin/bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 03:00:00
#SBATCH -J tb-niv
# usage: sbatch --array=0-26 run.sbatch ; task = job*9 + person; jobs: 0 new hardcbB, 1 new hardcbB_TB25, 2 ret hardcbB_TB25
# new: 8 full + 8 within seeds s71008..s71015, visits 1,2 ; ret: the R4 explicit orders (16 full + 16 within s71008..71023), visit 2, enrolled package
set -euo pipefail
R4=/leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120; T=/leonardo_work/AIFAC_P02_1636/ni26_laneTB_20261121; cd $R4
t=$SLURM_ARRAY_TASK_ID; j=$((t/9)); k=$((t%9))
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
export PYTHONPATH=$R4/stubs OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case $j in
 0) v=hardcbB; zip=zips/hardcbB.zip; mode=new;;
 1) v=hardcbB_TB25; zip=zips/hardcbB_TB25.zip; mode=new;;
 2) v=hardcbB_TB25; zip=enrolled/hardcbB_TB25_k$k.zip; mode=ret;;
esac
if [ $mode = new ]; then export NIV_VISITS=1,2; arms=$(python3 -c "print(','.join(f'{s}:s{71008+i}' for s in ('full','within') for i in range(8)))"); out=$T/out/new_${v}_k$k.npz
else export NIV_VISITS=2; arms=$(python3 -c "print(','.join(f'{s}:s{71008+i}' for s in ('full','within') for i in range(16)))"); mkdir -p out/${v}_k$k; out=out/${v}_k$k/ret.npz; fi
[ -e $out ] && { echo exists $out; exit 0; }
python -B niv_leo/worker.py $zip $out $k "$arms"
