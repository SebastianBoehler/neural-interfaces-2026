#!/bin/bash
set -euo pipefail
sha256sum *.py job.sbatch launch.sh >source.sha256
a=$(sbatch --parsable --time=4 job.sbatch inputsmoke)
b=$(sbatch --parsable --dependency=afterok:"$a" job.sbatch input)
c=$(sbatch --parsable --dependency=afterok:"$b" job.sbatch smoke whole)
d=$(sbatch --parsable --dependency=afterok:"$b" job.sbatch smoke half)
e=$(sbatch --parsable --array=0-9%4 --dependency=afterok:"$c" job.sbatch fit whole)
f=$(sbatch --parsable --array=0-9%4 --dependency=afterok:"$d" job.sbatch fit half)
g=$(sbatch --parsable --dependency=afterok:"$e":"$f" job.sbatch score)
echo "$a $b $c $d $e $f $g"
