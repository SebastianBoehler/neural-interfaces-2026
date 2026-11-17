#!/bin/bash
# Run on the authenticated workstation. Give each H100 checkpoint to the primary scorer.
set -euo pipefail
root=/mnt/lustre/work/gehler/gfh527
primary=$root/ni26_joint382_actor_v1_20261118
target=$root/ni26_joint382_h100_1382_20261118
states=$(ssh -o BatchMode=yes mlcloud-ferranti 'sacct -nPX -j 509169,509170,509171 -o JobID,State')
for id in 509169 509170 509171; do
  state=$(awk -F'|' -v id="$id" '$1==id {print $2}' <<< "$states")
  case "$state" in
    COMPLETED) ;;
    PENDING|RUNNING|CONFIGURING|COMPLETING|"") echo "Awaiting Ferranti $id ($state)"; exit 75 ;;
    *) echo "Ferranti job $id ended in $state" >&2; exit 1 ;;
  esac
done
if ssh -o BatchMode=yes mlcloud-galvani "test -s $target/actor/jobs.txt"; then
  ssh -o BatchMode=yes mlcloud-galvani "cat $target/actor/jobs.txt"
  exit 0
fi
ssh -o BatchMode=yes mlcloud-galvani "umask 077; mkdir -p $target/fits $target/actor"
ssh -o BatchMode=yes mlcloud-ferranti 'cd ~/ni26/382 && tar cf - F_1382.pt J_1382.pt K_1382.pt F_1382.json J_1382.json K_1382.json' |
  ssh -o BatchMode=yes mlcloud-galvani "cd $target/fits && tar xf -"
ssh -o BatchMode=yes mlcloud-galvani "set -euo pipefail; cd $target/actor; flock -n .launch.lock bash -ec '
  if test -s jobs.txt; then cat jobs.txt; exit; fi
  cd $primary && sha256sum -c source.sha256 >/dev/null
  cp actor382.py score382.py lock_fits.py actor.sbatch launch_actor.sh $target/actor/
  cd $target/actor
  bash launch_actor.sh $target/fits 1382 > jobs.txt
  cat jobs.txt'"
