#!/bin/bash
# Lane N queue (run in the NI-V release dir on the Galvani login node; only submits Slurm jobs).
#   bash queue.sh   -> score hardcbB_ttt{lite5,sparse20,light,heavy} (B-new, enrol, ret, fa), collectors vs cap250 and hardcbB,
#                      reference collectors (hard, cap250 vs hardcbB), summary job; all chained with --dependency.
set -euo pipefail
X="--exclude=galvani-cn122,galvani-cn116"
cd /mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119
all=""; cols=""
for f in lite5 sparse20 light heavy; do
  t=hardcbB_ttt$f
  [ -e zips/$t.zip ] || { echo "missing zips/$t.zip"; exit 1; }
  n=$(sbatch --parsable $X --array=0-35 score.sbatch $t.zip $t)
  e=$(sbatch --parsable enrol.sbatch $t.zip $t)
  r=$(sbatch --parsable $X --dependency=afterok:$e --array=0-8 enrol_run.sbatch $t ret)
  fa=$(sbatch --parsable $X --dependency=afterok:$e --array=0-8 enrol_run.sbatch $t fa)
  dep="afterok:$n:$r:$fa"
  c1=$(sbatch --parsable --dependency=$dep collect.sbatch $t cap250)
  c2=$(sbatch --parsable --dependency=$dep collect.sbatch $t hardcbB)
  echo "$t B-new $n enrol $e ret $r fa $fa collectors $c1 $c2"
  cols="$cols:$c1:$c2"
done
c3=$(sbatch --parsable collect.sbatch cap250 hardcbB)
sm=$(sbatch --parsable --dependency=afterany$cols:$c3 summary.sbatch)
echo "reference collector $c3  summary $sm (afterany, lists missing reports)"
