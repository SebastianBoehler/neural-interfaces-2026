#!/bin/bash
# Galvani login node, sbatch only. Per release (A, B) and tag: smoke -> B-new array (36) + enrolment (A only) -> B-ret array (9).
# Collectors (after everything of a release): nine25 vs hss and nine15 vs hss. The fa arm is not run.
set -euo pipefail
W=/mnt/lustre/work/gehler/gfh527; X="--exclude=galvani-cn122,galvani-cn116"
tags="hss hss_n25 hss_n15"
declare -A ENROL
for rel in A B; do
  cd $W/ni26_niv_final${rel}_20261121; all=""
  for t in $tags; do
    s=$(sbatch --parsable $X score.sbatch $t.zip $t smoke)
    n=$(sbatch --parsable $X --dependency=afterok:$s --array=0-35 score.sbatch $t.zip $t)
    if [ $rel = A ]; then
      e=$(sbatch --parsable enrol.sbatch $t.zip $t); ENROL[$t]=$e
    fi
    r=$(sbatch --parsable $X --dependency=afterok:${ENROL[$t]} --array=0-8 enrol_run.sbatch $t ret)
    all="$all:$n:$r"; echo "$rel $t smoke $s B-new $n enrol ${ENROL[$t]} ret $r"
  done
  for t in hss_n25 hss_n15; do
    c=$(sbatch --parsable --dependency=afterany${all} collect.sbatch $t hss); echo "$rel collector $t vs hss $c"
  done
done
