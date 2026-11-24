#!/bin/bash
# Lane E on Leonardo (use when eduVPN is OFF and the step certificate is fresh). Run from the repo root:
#   lane_e/leo_release.sh ZIP_DIR [ACCOUNT=AIFAC_P02_1636]
# ZIP_DIR holds E0 Ea Eb Ec Ed Ed_g2 (+ the signed cap250 archive is taken from runs/sealed/final_zips). Everything is scored on
# A100 so the reference is re-scored there too (cap250, E0 as the same-hardware null: it must equal cap250 bit for bit; its delta
# calibrates run-to-run noise). Submits only Slurm jobs. Collectors run on the cluster and wait on their arrays.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(dirname "$here"); Z=${1:?}; ACC=${2:-AIFAC_P02_1636}
H=cineca-leonardo; D=${LEO_D:-/leonardo_work/AIFAC_P02_1636/ni26_lane_e_capacity}
ROWS=${SCHERER_ROWS:-/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/340c1963-08fd-48fa-8913-a19f30c7cf3e/scratchpad/scherer_rows.tsv.gz}
for a in E0 Ea Eb Ec Ed Ed_g2; do test -f "$Z/$a.zip"; done; test -f "$ROWS"
ssh $H "mkdir -p $D/zips $D/data $D/code $D/logs $D/out $D/niv_leo"
rsync -a --partial "$Z"/*.zip "$repo/runs/sealed/final_zips/v66_cp1_cap250.zip" $H:$D/zips/
rsync -a --partial "$repo/data/scherer2015/windows.npz" "$ROWS" "$repo/runs/niv/cohort_a.npz" $H:$D/data/
rsync -a --exclude __pycache__ "$here" $H:$D/code/
rsync -a "$here"/leo_*.sbatch $H:$D/
rsync -a --exclude __pycache__ "$repo"/niv/{orders,bank,stats,collect,worker}.py $H:$D/niv_leo/
ssh $H "cd $D && bash code/lane_e/leo_submit.sh $ACC"
