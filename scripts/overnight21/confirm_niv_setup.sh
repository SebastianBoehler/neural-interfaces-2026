#!/bin/bash
# Galvani, file operations only: two NI-V release dirs for the hard_safe_st / +nine25 / +nine15 exact packages.
#   A = orders 71000-71007 (v1 seeds), B = orders 71008-71015 (as ni26_niv_x8_20261121). Packages are symlinks to the pinned ZIPs
#   in ni26_final_confirm_20261121/zips; enrolled packages of B are A's (enrolment does not depend on the order seeds).
set -euo pipefail
W=/mnt/lustre/work/gehler/gfh527; V=$W/ni26_niv_v1_20261119; Z=$W/ni26_final_confirm_20261121/zips
declare -A SRC=([hss]=final_stack_hard_safe_st [hss_n25]=final_stack_hard_safe_st_team_b_nine25_tbgraph [hss_n15]=final_stack_hard_safe_st_team_b_nine15_tbgraph)
for rel in A B; do
  X=$W/ni26_niv_final${rel}_20261121; mkdir -p $X/logs $X/out $X/zips; cd $X
  for f in collect.sbatch enrol.sbatch enrol_run.sbatch score.sbatch scherer_rows.tsv.gz cohort_a.npz; do cp $V/$f .; done
  cp -r $V/niv $V/studies $V/replay .; rm -rf niv/__pycache__
  grep -v ' zips/' $V/inputs.sha256 > inputs.sha256
  for t in "${!SRC[@]}"; do ln -sfn $Z/${SRC[$t]}.zip zips/$t.zip; sha256sum zips/$t.zip >> inputs.sha256; done
  if [ $rel = B ]; then
    sed -i 's/SEEDS = tuple(71000 + k for k in range(8))/SEEDS = tuple(71008 + k for k in range(8))/' niv/orders.py
    ln -sfn $W/ni26_niv_finalA_20261121/zips/enrolled zips/enrolled
    # inputs.sha256 pins niv/orders.py of v1; B changes it by design
    grep -v ' niv/orders.py' inputs.sha256 > i2 && mv i2 inputs.sha256; sha256sum niv/orders.py >> inputs.sha256
    python3 -c "import sys; sys.path.insert(0,'niv'); import orders; assert orders.SEEDS == tuple(range(71008,71016)), orders.SEEDS"
  else
    mkdir -p zips/enrolled
    # the zips are symlinks into ni26_final_confirm_20261121/zips: enrol.sbatch must bind that directory (it failed with FileNotFoundError otherwise)
    sed -i "s#--bind \"\$work:/work:rw,#--bind \"\$work:/work:rw,$Z:$Z:ro,#" enrol.sbatch; grep -c "$Z:$Z:ro" enrol.sbatch
  fi
  sha256sum -c inputs.sha256 | grep -vc OK || true
  echo "release $rel ready: $X"
done
