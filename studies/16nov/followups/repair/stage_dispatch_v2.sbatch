#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-302-stage-queue
#SBATCH --output=logs/302-stage-queue-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes302_20261116/repair
sha256sum -c repair-stage.sha256
sha256sum -c repair-stage-dispatch-v2.sha256
[[ "$(sacct -X -n -P -j 3114266 -o State,ExitCode | head -1)" == 'COMPLETED|0:0' ]]
score=$(awk '$1 == "score" {print $2}' runs/lane1_inputs/repair302.intent)
[[ "$score" =~ ^[0-9]+$ ]] || { echo 'missing full-score job'; exit 1; }
[[ ! -e runs/lane1_inputs/repair302-stage.intent ]] || { echo 'stage dispatch already attempted'; exit 1; }
( set -o noclobber; printf 'dispatching after score %s\n' "$score" > runs/lane1_inputs/repair302-stage.intent )
if [[ -n "$(squeue -h -j "$score" -o %T)" ]]; then
  id=$(sbatch --parsable --dependency="afterok:$score" studies/16nov/followups/repair/stage.sbatch)
else
  [[ "$(sacct -X -n -P -j "$score" -o State,ExitCode | head -1)" == 'COMPLETED|0:0' ]]
  grep -q '§302 exploratory full-system repair' "logs/302-score-$score.out"
  id=$(sbatch --parsable studies/16nov/followups/repair/stage.sbatch)
fi
printf 'stage %s after score %s\n' "$id" "$score" >> runs/lane1_inputs/repair302-stage.intent
printf 'stage %s after score %s\n' "$id" "$score"
