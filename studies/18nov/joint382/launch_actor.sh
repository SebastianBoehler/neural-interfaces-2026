#!/bin/bash
set -euo pipefail
root=${1:?absolute fit path}
seed=${2:?registered seed}
python=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
digest=$("$python" lock_fits.py "$root" "$seed")
sha256sum actor382.py score382.py lock_fits.py actor.sbatch launch_actor.sh fit_lock.json > source.sha256
a=$(sbatch --parsable --array=0 actor.sbatch actor "$root" "$seed" "$digest")
b=$(sbatch --parsable --dependency=afterok:"$a" --array=1-3 actor.sbatch actor "$root" "$seed" "$digest")
c=$(sbatch --parsable --dependency=afterok:"$a":"$b" actor.sbatch score "$root" "$seed" "$digest")
echo "$a $b $c $digest"
