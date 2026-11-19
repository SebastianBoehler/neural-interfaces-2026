#!/bin/bash
# Stage the gS fold-package release on Galvani and submit: fold fit -> smoke -> 32-task array -> collector (run locally).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(dirname "$here")
base=/mnt/lustre/work/gehler/gfh527; dest=$base/ni26_laneG_gs_returning_20261119
ssh mlcloud-galvani "mkdir -p $dest/logs; cp $base/ni26_lane_a_returning_20261119/{lane_inputs.py,dose_worker.py,parents.sha256} $dest/"
rsync -a "$here"/{common.py,fit.py,gs_fold.py,gs_fold.sbatch,gs_worker.py,gs_worker.sbatch,gs_collect.py,gs_collect.sbatch} mlcloud-galvani:$dest/
ssh mlcloud-galvani "cd $dest && sha256sum lane_inputs.py dose_worker.py gs_worker.py > source.sha256"
