#!/bin/bash
# Galvani, file operations only: NI-V release C = orders 71200-71207 for the hard_safe_st / +nine25 / +nine15 exact packages.
# Built from release B (same ZIP symlinks); enrolled packages are A's (enrolment does not depend on the order seeds).
set -euo pipefail
W=/mnt/lustre/work/gehler/gfh527; B=$W/ni26_niv_finalB_20261121; X=$W/ni26_niv_finalC_20261121
mkdir -p $X/logs $X/out $X/zips; cd $X
for f in collect.sbatch enrol.sbatch enrol_run.sbatch score.sbatch scherer_rows.tsv.gz cohort_a.npz inputs.sha256; do cp $B/$f .; done
cp -r $B/niv $B/studies $B/replay .; rm -rf niv/__pycache__
for t in hss hss_n25 hss_n15; do ln -sfn $(readlink -f $B/zips/$t.zip) zips/$t.zip; done
ln -sfn $W/ni26_niv_finalA_20261121/zips/enrolled zips/enrolled
sed -i 's/SEEDS = tuple(71008 + k for k in range(8))/SEEDS = tuple(71200 + k for k in range(8))/' niv/orders.py
python3 -c "import sys; sys.path.insert(0,'niv'); import orders; assert orders.SEEDS == tuple(range(71200,71208)), orders.SEEDS"
grep -v ' niv/orders.py' inputs.sha256 > i2 && mv i2 inputs.sha256; sha256sum niv/orders.py >> inputs.sha256
sha256sum -c inputs.sha256 | grep -vc OK || true
echo "release C ready"
