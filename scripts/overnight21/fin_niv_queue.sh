#!/bin/bash
# Galvani login node, sbatch only. Release C (orders 71200-71207): per tag smoke -> B-new array (36) and B-ret array (9, A's enrolment
# jobs are done, enrolled packages are linked), then collectors n25 vs hss and n15 vs hss. Stay under the 300-task cap.
set -euo pipefail
W=/mnt/lustre/work/gehler/gfh527; X="--exclude=galvani-cn122,galvani-cn116"
cd $W/ni26_niv_finalC_20261121; all=""
for t in hss hss_n25 hss_n15; do
  s=$(sbatch --parsable $X score.sbatch $t.zip $t smoke)
  n=$(sbatch --parsable $X --dependency=afterok:$s --array=0-35 score.sbatch $t.zip $t)
  r=$(sbatch --parsable $X --dependency=afterok:$s --array=0-8 enrol_run.sbatch $t ret)
  all="$all:$n:$r"; echo "C $t smoke $s B-new $n ret $r"
done
for t in hss_n25 hss_n15; do echo "C collector $t vs hss $(sbatch --parsable --dependency=afterany$all collect.sbatch $t hss)"; done
