#!/bin/bash
# s290 part 2 (run on Galvani, only sbatch; each line in its own work dir). Tasks 0-11 = 4 cells x 3 seeds per family, 2080 partitions, lanes %3 each. Usage: submit2.sh r2|r3|r4a|r4b
set -euo pipefail
export PART=${PART:-2080-galvani,2080-preemptable-galvani}
G=/mnt/lustre/work/gehler/gfh527
case $1 in
r2) cd $G/ni26_grid283 && scripts/cloud/galv_array.sh g290r2 06:00:00 "FAMS=fbl_coral10_am50 bash studies/15nov/grid283/cell.sh" "0-11%3";;  # CORAL10 + B1 (p 0.5) on fbl
r3) cd $G/ni26_grid283 && DEP=${DEP:---dependency=afterok:3107981} scripts/cloud/galv_array.sh g290r3 06:00:00 "FAMS=fbw2_dist2 bash studies/15nov/s290/cell.sh" "0-11%3";;  # width x2 student, T 2, after the REVE teachers
r4a) cd $G/ni26_tsm288 && scripts/cloud/galv_array.sh g290tsmlo 06:00:00 "FAMS=tsm_lr3e4 bash studies/15nov/s290/cell.sh" "0-11%3";;
r4b) cd $G/ni26_tsm288 && scripts/cloud/galv_array.sh g290tsmhi 06:00:00 "FAMS=tsm_lr3e3 bash studies/15nov/s290/cell.sh" "0-11%3";;
esac
