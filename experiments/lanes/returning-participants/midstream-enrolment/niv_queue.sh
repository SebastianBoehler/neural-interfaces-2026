#!/bin/bash
# Lane R4 NI-V queue (run in the NI-V release dir on the Galvani login node; only submits Slurm jobs).
#   R4_RET=/mnt/lustre/work/gehler/gfh527/ni26_laneR4_20261120/ret bash niv_queue.sh R1 R2 ...   (zips already built by ret_build.sbatch)
# per arm: copy+pin job, B-new (36), enrol, ret (9), fa (9), collectors vs hardcbB and cap250; summary afterany. Tags hardcbB_<arm>.
set -euo pipefail
X="--exclude=galvani-cn122,galvani-cn116"
cd /mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119
src=${R4_RET:?}/zips
cp1=$(sbatch --parsable --partition=cpu-galvani --mem=2G --time=00:10:00 -J ni26-R4-cp -o logs/R4cp-%j.out --wrap="set -e; cd $PWD; for a in $*; do cp -n $src/hardcbB_\$a.zip zips/hardcbB_\$a.zip; sha256sum zips/hardcbB_\$a.zip >> inputs.sha256; done")
cols=""
for a in "$@"; do
  t=hardcbB_$a
  n=$(sbatch --parsable $X --dependency=afterok:$cp1 --array=0-35 score.sbatch $t.zip $t)
  e=$(sbatch --parsable --dependency=afterok:$cp1 enrol.sbatch $t.zip $t)
  r=$(sbatch --parsable $X --dependency=afterok:$e --array=0-8 enrol_run.sbatch $t ret)
  fa=$(sbatch --parsable $X --dependency=afterok:$e --array=0-8 enrol_run.sbatch $t fa)
  dep="afterok:$n:$r:$fa"
  c1=$(sbatch --parsable --dependency=$dep collect.sbatch $t hardcbB)
  c2=$(sbatch --parsable --dependency=$dep collect.sbatch $t cap250)
  echo "$t B-new $n enrol $e ret $r fa $fa collectors $c1 $c2"
  cols="$cols:$c1:$c2"
done
echo "copy $cp1; collectors:$cols"
