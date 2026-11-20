#!/bin/bash
#SBATCH --account=AIFAC_P02_1675
#SBATCH -p lrd_all_serial
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J r4b-setup
# builds the R4b release: copy of the B10 release (code, no outputs), ext7 (ds007221g), all dose / gated packages from hardcbB.zip. usage: sbatch laneR4b/setup.sbatch (from the new release root, code already rsynced)
set -euo pipefail
W=/leonardo_work/AIFAC_P02_1675; OLD=$W/ni26_laneB10_20261120; R=$W/ni26_laneR4b_20261120; cd $R
for f in $(ls $OLD | grep -v -e '^out$' -e '^logs$' -e '^slurm' -e '^laneB10$'); do [ -e $R/$f ] || ln -s $OLD/$f $R/$f; done
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
Z=/leonardo_work/AIFAC_P02_1636/ni26_laneR4_20261120/zips
mkdir -p ext7 zips logs out
python -B laneR4b/prep7.py ext7 $W/ni26_laneB6_20261120/data/ds007221g/windows_*.npz
for a in R4n R4s2 R4s4 R4t2 R4h1 Gconf_on Gconf_off; do [ -e zips/$a.zip ] || python -B laneR4b/build_dose.py $a zips/$a.zip $Z/hardcbB.zip; done
for a in hardcbB hardcbB_R1a hardcbB_R2 hardcbB_R3 hardcbB_R4 hardcbB_R5; do ln -sf $Z/$a.zip zips/$a.zip; done
# base consistency: the R4 text built here must equal the R4 package of laneR4 (same anchors, same source)
python - <<'PY'
import zipfile
a = zipfile.ZipFile("zips/Gconf_on.zip").read("submission.py").decode()
b = zipfile.ZipFile("zips/hardcbB_R4.zip").read("submission.py").decode()
assert "lane R4b" in a and a.count("GATE_THETA") > 3
import difflib
d = [l for l in difflib.unified_diff(b.splitlines(), a.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
print("diff lines vs R4 package", len(d))
PY
ls zips
