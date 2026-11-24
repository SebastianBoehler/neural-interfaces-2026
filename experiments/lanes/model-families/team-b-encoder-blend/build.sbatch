#!/bin/bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=30G
#SBATCH -t 00:30:00
#SBATCH -J tb-build
# builds zips/hardcbB_TB{25,0}.zip, then enrolled/hardcbB_TB25_k{0..8}.zip
set -euo pipefail
R4=/leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120; T=/leonardo_work/AIFAC_P02_1636/ni26_laneTB_20261121
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
mkdir -p $T/zips; cd $T
[ -e zips/hardcbB_TB25.zip ] || python -B build_niv.py $R4/zips/hardcbB.zip /leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/laneTB-21nov/package/weights.pt zips
cd $R4; cp -n $T/zips/hardcbB_TB*.zip zips/
export PYTHONPATH=$R4/stubs OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
python -B niv_leo/build_enrolled.py zips/hardcbB_TB25.zip enrolled hardcbB_TB25
sha256sum zips/hardcbB_TB*.zip
