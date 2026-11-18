#!/bin/bash
# Run on the workstation after Ferranti completes all six source-only fits.
set -euo pipefail
ferranti=mlcloud-ferranti
galvani=mlcloud-galvani
root=/mnt/lustre/work/gehler/gfh527/ni26_conflict388_v1_20261118
states=$(ssh "$ferranti" 'sacct -nPX -j 509377 -o jobid,state')
for i in {0..5}; do
  status=$(awk -F'|' -v id="509377_$i" '$1==id {print $2}' <<<"$states")
  if [[ "$status" != COMPLETED ]]; then
    echo "Ferranti 509377_$i is ${status:-pending}" >&2
    exit 75
  fi
done
ssh "$ferranti" 'cd ~/ni26/388 && tar cf - N_382.{json,pt} H_382.{json,pt} X_382.{json,pt} N_1382.{json,pt} H_1382.{json,pt} X_1382.{json,pt}' |
  ssh "$galvani" "cd $root/fits && tar xf -"
ssh "$galvani" "cd $root && bash -s" <<'REMOTE'
set -euo pipefail
if [[ -f jobs.txt ]]; then cat jobs.txt; exit; fi
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$py" -B verify388.py
sha256sum actor388.py score388.py verify388.py actor.sbatch fits.json >source.sha256
a=$(sbatch --parsable --array=0 actor.sbatch actor)
b=$(sbatch --parsable --dependency=afterok:"$a" --array=1-3 actor.sbatch actor)
c=$(sbatch --parsable --dependency=afterok:"$b" actor.sbatch score)
echo "$a $b $c" |tee jobs.txt
REMOTE
