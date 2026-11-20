#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH -p boost_usr_prod
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -J b10-prep
# usage (release root): sbatch laneB10/prep.sbatch -> ext/p{0..18}.npz
set -euo pipefail
cd /leonardo_work/AIFAC_P02_1675/ni26_laneB10_20261120
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
D=/leonardo_work/AIFAC_P02_1675/ni26_laneB6_20261120/data/netbci
python -B laneB10/prep.py ext $D/windows_01-10.npz $D/windows_11-19.npz
