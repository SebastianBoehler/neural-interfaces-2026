#!/bin/bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 00:40:00
#SBATCH -J tb-smoke
# one person (k=5), one order (full:s71008), visit 1 only, first 150 windows are NOT cut by the worker, so the whole visit is run: hardcbB vs TB0 vs TB25
set -euo pipefail
R4=/leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120; T=/leonardo_work/AIFAC_P02_1636/ni26_laneTB_20261121; cd $R4
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
export PYTHONPATH=$R4/stubs OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NIV_VISITS=1
mkdir -p $T/out/smoke
for v in hardcbB hardcbB_TB0 hardcbB_TB25; do python -B niv_leo/worker.py zips/$v.zip $T/out/smoke/$v.npz 5 full:s71008 & done; wait
python - <<PY
import numpy as np
d={v:np.load(f"$T/out/smoke/{v}.npz") for v in ("hardcbB","hardcbB_TB0","hardcbB_TB25")}
key=[k for k in d["hardcbB"].files if k.endswith("|labels")][0]
a,b,c=(d[v][key] for v in d)
print("n",len(a),"TB0==base",bool((a==b).all()),"TB25 differs",int((a!=c).sum()),"of",len(a))
assert (a==b).all() and (a!=c).sum()>0
PY
