#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH -p lrd_all_serial
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -J r4b-gated
# gated packages of wave 2 (thetas fixed in evidence/wave_20nov/laneR4b_dose.md before any gated arm was scored). usage: sbatch laneR4b/build_gated.sbatch from the release root
set -euo pipefail
R=/leonardo_work/AIFAC_P02_1675/ni26_laneR4b_20261120; cd $R
source /leonardo_work/AIFAC_P02_1675/../AIFAC_P02_1636/ni26/.venv/bin/activate
for a in Gconf_0.59 Gconf_0.62 Gconf_0.65 Gconf_0.70 Gagree_0.55 Gagree_0.65 Gent_0.25 Gent_0.32; do [ -e zips/$a.zip ] || python -B laneR4b/build_dose.py $a zips/$a.zip /leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120/zips/hardcbB.zip; done
ls zips
