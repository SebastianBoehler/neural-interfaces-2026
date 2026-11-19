#!/bin/bash
# Run after the D1 smoke collector says launch. Start from the staged Leonardo repo.
set -euo pipefail
account=${1:?expected an authorized Leonardo Slurm account}
work=${2:?expected the absolute D1 artifact directory}
checkpoint=${3:?expected the absolute pinned REVE backbone path}
[[ "$work" = /* && "$checkpoint" = /* && "$PWD" != *" "* && "$work" != *" "* && "$checkpoint" != *" "* ]]
test -d "$work"
test -s "$checkpoint"
test -s "$work/reve_positions.json"
grep -q '"decision": "launch"' "$work/smoke_result.json"
grep -q '8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe' laneC/model.py
mkdir -p runs/diag/arr
root=$PWD
sbatch_id() { sbatch --parsable "$@" | cut -d ';' -f1; }
common="module load python/3.11.7; source .venv/bin/activate; export HF_HUB_OFFLINE=1; export REVE_POSITIONS_PATH='$work'; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
canary=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=60G -t 00:30:00 \
  -J ni26-d1-canary -o runs/diag/arr/d1-canary-%j.out --wrap "set -e; $common; python laneC/train.py canary-b 0 --root '$root' --work '$work' --checkpoint '$checkpoint'")
# Three fresh stage-A seeds feed 10 LOPO folds, one returning fit, and one all-public fit each.
extern=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=60G -t 01:30:00 \
  --dependency=afterok:"$canary" --kill-on-invalid-dep=yes --array=0-2%3 \
  -J ni26-d1-external -o runs/diag/arr/d1-external-%A_%a.out \
  --wrap "set -e; $common; python laneC/train.py external-full \$SLURM_ARRAY_TASK_ID --root '$root' --work '$work' --checkpoint '$checkpoint'")
fit=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=60G -t 01:30:00 \
  --dependency=afterok:"$extern" --kill-on-invalid-dep=yes --array=0-35%12 \
  -J ni26-d1-fit -o runs/diag/arr/d1-fit-%A_%a.out \
  --wrap "set -e; $common; python laneC/train.py full \$SLURM_ARRAY_TASK_ID --root '$root' --work '$work' --checkpoint '$checkpoint'")
collector=$(DEP="--dependency=afterok:$fit --kill-on-invalid-dep=yes" \
  bash scripts/cloud/leo_cpu.sh d1-standalone "$account" 00:15:00 "python laneC/full_collect.py '$root' '$work'")
printf '%s\n' "canary=$canary external=$extern full=$fit collector=$collector"
