#!/bin/bash
# Stage the Lane A release on Galvani: scripts, hash files, and dosed fold packages (symlinks + dosed submission.py).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(dirname "$here")
base=/mnt/lustre/work/gehler/gfh527; dest=$base/ni26_lane_a_returning_20261119
inner=$base/ni26_friday_returning_native_inner_v2_20261119
bundle=$base/ni26_js321_20261117_returning_bundle2
ssh mlcloud-galvani "mkdir -p $dest/pkg"
rsync -a "$here"/lane_inputs.py "$here"/dose_worker.py "$here"/dose_worker.sbatch "$here"/collect.py "$here"/collect.sbatch mlcloud-galvani:$dest/
rsync -a "$repo"/submissions/lean_v66_dose/submission.py mlcloud-galvani:$dest/dose_submission.py
ssh mlcloud-galvani "cd $dest && for f in P04 P10; do d=pkg/\${f}_s23; rm -rf \$d; mkdir -p \$d; \
  for g in $bundle/\${f}_s23/base/*; do n=\$(basename \$g); [ \$n = submission.py ] || ln -s \$g \$d/\$n; done; \
  cp dose_submission.py \$d/submission.py; done; \
  (cd $inner && sha256sum worker.py inputs.py assets.py trace_inputs.py) > parents.sha256; \
  sha256sum lane_inputs.py dose_worker.py dose_worker.sbatch collect.py dose_submission.py > source.sha256; ls pkg/P04_s23 | head -3; cat parents.sha256"
