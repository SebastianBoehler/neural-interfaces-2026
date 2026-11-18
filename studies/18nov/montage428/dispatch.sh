#!/bin/bash
# Precondition: the first real-person build was submitted separately.
set -eu
cpu=(-A gehler -p a100-galvani,2080-galvani -c 2 --mem=10G -t 12 --kill-on-invalid-dep=yes)
gpu=(-A gehler -p a100-galvani,a100-preemptable-galvani --gres=gpu:a100:1 -c 2 --mem=16G -t 12 --kill-on-invalid-dep=yes)
py=../neural-interfaces-2026/.venv/bin/python
prep=$(sbatch --parsable "${cpu[@]}" --dependency="afterok:$1" --array=1-8%3 -J ni26-m428p --wrap="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $py build.py \$SLURM_ARRAY_TASK_ID")
lane=$(sbatch --parsable "${cpu[@]}" --dependency="afterok:$prep" -J ni26-m428l --wrap="$py build.py assemble && $py lanes.py ../ni26_cross411b_20261118 lane local.npz")
smoke=$(sbatch --parsable "${gpu[@]}" --dependency="afterok:$lane" -J ni26-m428sm --wrap="cd lane && bash launch.sbatch")
array=$(sbatch --parsable "${gpu[@]}" --array=0-8%2 --dependency="afterok:$smoke" -J ni26-m428 --wrap="cd lane && bash launch.sbatch")
score=$(sbatch --parsable "${cpu[@]}" --dependency="afterok:$array" -J ni26-m428s --wrap="(cd lane && bash collect.sbatch) && $py score.py")
echo "$prep $lane $smoke $array $score"
