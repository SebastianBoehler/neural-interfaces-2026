#!/bin/bash
# Second pass for release B after the 300-task cap stopped the first pass: hss_n15 (smoke -> B-new -> B-ret on A's enrolment 3145728)
# and the two collectors (afterany on every B job of the first pass 3145733/34/36/37 and this pass).
set -euo pipefail
W=/mnt/lustre/work/gehler/gfh527; X="--exclude=galvani-cn122,galvani-cn116"
cd $W/ni26_niv_finalB_20261121
s=${1:-$(sbatch --parsable $X score.sbatch hss_n15.zip hss_n15 smoke)}  # pass the id of an already submitted smoke to reuse it
until [ "$(squeue -u $USER -h -r | wc -l)" -le 250 ]; do sleep 60; done  # 300-task cap
n=$(sbatch --parsable $X --dependency=afterok:$s --array=0-35 score.sbatch hss_n15.zip hss_n15)
r=$(sbatch --parsable $X --dependency=afterok:3145728 --array=0-8 enrol_run.sbatch hss_n15 ret)
all=":3145733:3145734:3145736:3145737:$n:$r"
echo "B hss_n15 smoke $s B-new $n ret $r"
for t in hss_n25 hss_n15; do echo "B collector $t vs hss $(sbatch --parsable --dependency=afterany$all collect.sbatch $t hss)"; done
