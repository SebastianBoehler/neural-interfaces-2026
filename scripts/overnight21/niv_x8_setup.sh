#!/bin/bash
# Galvani (run as a Slurm job): release copy of NI-V v1 whose orders are 71008-71015 (instead of 71000-71007).
# zips and enrolled packages are symlinks to the pinned originals; niv/orders.py is the only changed code file.
set -euo pipefail
R=/mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119; X=/mnt/lustre/work/gehler/gfh527/ni26_niv_x8_20261121
mkdir -p $X/logs $X/out $X/zips; cd $X
for f in collect.sbatch enrol_run.sbatch score.sbatch scherer_rows.tsv.gz; do cp $R/$f .; done
grep -E ' zips/(v66_cp1_cap250|hardcbB)\.zip$' $R/inputs.sha256 | sort -u > inputs.sha256; test $(wc -l < inputs.sha256) -eq 2
cp -r $R/niv niv; rm -rf niv/__pycache__
for z in v66_cp1_cap250.zip hardcbB.zip; do ln -sfn $R/zips/$z zips/$z; done
ln -sfn $R/zips/enrolled zips/enrolled
sed -i 's/SEEDS = tuple(71000 + k for k in range(8))/SEEDS = tuple(71008 + k for k in range(8))/' niv/orders.py
python3 -c "
import sys; sys.path.insert(0,'niv'); import orders
assert orders.SEEDS == tuple(range(71008,71016)), orders.SEEDS"
grep -c "71008" niv/orders.py; sha256sum -c inputs.sha256; echo setup-ok
