#!/bin/bash
# From the immutable study directory, pass the crop assembler job ID.
set -euo pipefail
prep=$(sbatch --parsable -A gehler -p a100-galvani,2080-galvani -c 1 --mem=2G -t 4 -J ni26-cue425l \
  --dependency="afterok:$1" --kill-on-invalid-dep=yes --wrap="../neural-interfaces-2026/.venv/bin/python lanes.py")
cpu=(-A gehler -p a100-galvani,2080-galvani -c 2 --mem=6G -t 8 --kill-on-invalid-dep=yes)
gpu=(-A gehler -p a100-galvani,a100-preemptable-galvani --gres=gpu:a100:1 -c 2 --mem=16G -t 12 --kill-on-invalid-dep=yes)
scorers=()
echo "lane setup $prep"
for offset in 30 60; do
    launch="cd offset$offset && bash launch.sbatch"
    s=$(sbatch --parsable "${gpu[@]}" --dependency="afterok:$prep" -J ni26-cue425-smoke --wrap="$launch")
    a=$(sbatch --parsable "${gpu[@]}" --array=0-8%2 --dependency="afterok:$s" -J ni26-cue425 --wrap="$launch")
    c=$(sbatch --parsable "${cpu[@]}" --dependency="afterok:$a" -J ni26-cue425score --wrap="cd offset$offset && bash collect.sbatch")
    scorers+=("$c")
    echo "$offset $s $a $c"
done
sbatch --parsable "${cpu[@]}" --dependency="afterok:${scorers[0]}:${scorers[1]}" -J ni26-cue425final \
    --wrap="../neural-interfaces-2026/.venv/bin/python score.py"
