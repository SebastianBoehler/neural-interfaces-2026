#!/bin/bash
# Galvani, file operations only: NI-V release REL for orders START..START+7 (hss / hss_n25 / hss_n15 exact packages, A's enrolled packages).
# Usage: fin_niv_setup_gen.sh REL START   (copies release C, rewrites SEEDS, re-pins niv/orders.py)
set -euo pipefail
rel=${1:?}; s=${2:?}
W=/mnt/lustre/work/gehler/gfh527; C=$W/ni26_niv_finalC_20261121; X=$W/ni26_niv_final${rel}_20261121
[ -e $X ] && { echo "exists $X"; exit 1; }
mkdir -p $X/logs $X/out $X/zips; cd $X
for f in collect.sbatch enrol.sbatch enrol_run.sbatch score.sbatch scherer_rows.tsv.gz cohort_a.npz inputs.sha256; do cp $C/$f .; done
cp -r $C/niv $C/studies $C/replay .; rm -rf niv/__pycache__
for t in hss hss_n25 hss_n15; do ln -sfn $(readlink -f $C/zips/$t.zip) zips/$t.zip; done
ln -sfn $W/ni26_niv_finalA_20261121/zips/enrolled zips/enrolled
sed -i "s/SEEDS = tuple(71200 + k for k in range(8))/SEEDS = tuple($s + k for k in range(8))/" niv/orders.py
python3 -c "import sys; sys.path.insert(0,'niv'); import orders; assert orders.SEEDS == tuple(range($s,$s+8)), orders.SEEDS"
grep -v ' niv/orders.py' inputs.sha256 > i2 && mv i2 inputs.sha256; sha256sum niv/orders.py >> inputs.sha256
sha256sum -c inputs.sha256 | grep -vc OK || true
echo "release $rel ready (orders $s-$((s+7)))"
