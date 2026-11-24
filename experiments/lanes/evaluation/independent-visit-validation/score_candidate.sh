#!/bin/bash
# Score a candidate ZIP on NI-V (run on the Galvani login node in the release dir; only submits Slurm jobs).
#   cd /mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119
#   cp /path/to/candidate.zip zips/CAND.zip
#   bash score_candidate.sh CAND [REF=cap250] [NULL_TAG]
# B-new arm (36 tasks), enrolment of the 9 Scherer people (visit 1), returning + zero-routing arms, and an
# on-cluster collector writing out/report_CAND_vs_REF.json (paired delta vs the reference tag, full-session headline).
set -euo pipefail
cand=${1:?}; ref=${2:-cap250}; null=${3:-}
X="--exclude=galvani-cn122,galvani-cn116"
[ -e zips/$cand.zip ] || { echo "zips/$cand.zip missing"; exit 1; }
sha256sum zips/$cand.zip >> inputs.sha256   # pin the candidate (all earlier lines stay valid)
n=$(sbatch --parsable $X --array=0-35 score.sbatch $cand.zip $cand)
e=$(sbatch --parsable enrol.sbatch $cand.zip $cand)
r=$(sbatch --parsable $X --dependency=afterok:$e --array=0-8 enrol_run.sbatch $cand ret)
f=$(sbatch --parsable $X --dependency=afterok:$e --array=0-8 enrol_run.sbatch $cand fa)
c=$(sbatch --parsable --dependency=afterany:$n:$r:$f collect.sbatch $cand $ref $null)
echo "B-new $n  enrol $e  ret $r  fa $f  collector $c  -> out/report_${cand}_vs_${ref}.json"
