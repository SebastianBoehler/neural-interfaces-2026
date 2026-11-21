#!/bin/bash
# Slurm-side or login-node sbatch only. Usage: fin_niv_queue_gen.sh REL. Per tag smoke -> B-new (36) + B-ret (9), then collectors vs hss.
set -euo pipefail
rel=${1:?}; W=/mnt/lustre/work/gehler/gfh527; X="--exclude=galvani-cn122,galvani-cn116"
cd $W/ni26_niv_final${rel}_20261121; all=""
for t in hss hss_n25 hss_n15; do
  s=$(sbatch --parsable $X score.sbatch $t.zip $t smoke)
  n=$(sbatch --parsable $X --dependency=afterok:$s --array=0-35 score.sbatch $t.zip $t)
  r=$(sbatch --parsable $X --dependency=afterok:$s --array=0-8 enrol_run.sbatch $t ret)
  all="$all:$n:$r"; echo "$rel $t smoke $s B-new $n ret $r"
done
for t in hss_n25 hss_n15; do echo "$rel collector $t vs hss $(sbatch --parsable --dependency=afterany$all collect.sbatch $t hss)"; done
