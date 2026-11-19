#!/bin/bash
# Requires the pinned checkpoint, position file and staged data root. Submit from the release root.
set -euo pipefail
account=${1:?expected an authorized Leonardo Slurm account}
work=${2:?expected the absolute D1 artifact directory}
checkpoint=${3:?expected the absolute pinned REVE backbone path}
[[ "$work" = /* && "$checkpoint" = /* && "$PWD" != *" "* && "$work" != *" "* && "$checkpoint" != *" "* ]]
test -d "$work"
test -s "$checkpoint"
test -s "$work/reve_positions.json"
mkdir -p logs
root=$PWD
sbatch_id() { sbatch --parsable "$@" | cut -d ';' -f1; }
common="module load python/3.11.7; unset PYTHONPATH; source .venv/bin/activate; export HF_HUB_OFFLINE=1 REVE_POSITIONS_PATH='$work'; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; sha256sum -c actors.sha256 --quiet"
actor="python -u laneC/train.py"
args="--root '$root' --work '$work' --checkpoint '$checkpoint'"
ready='for ((attempt=0; attempt<48; attempt++)); do test -s data/shin2018/windows.npz && break; sleep 15; done; sha256sum --quiet -c staged_data.sha256'
canary=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t 00:25:00 \
  -J ni26-d1-canary -o logs/d1-canary-%j.out --wrap "set -e; $ready; $common; $actor canary 0 $args")
extern=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t 01:15:00 \
  --dependency=afterok:"$canary" --kill-on-invalid-dep=yes --array=0-5%6 \
  -J ni26-d1-external -o logs/d1-external-%A_%a.out \
  --wrap "set -e; $common; $actor external-full \$SLURM_ARRAY_TASK_ID $args")
fit=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t 01:30:00 \
  --dependency=afterok:"$extern" --kill-on-invalid-dep=yes --array=0-71%20 \
  -J ni26-d1-fit -o logs/d1-fit-%A_%a.out \
  --wrap "set -e; $common; $actor full \$SLURM_ARRAY_TASK_ID $args")
collector=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 2 --mem=16G -t 00:30:00 \
  --dependency=afterok:"$fit" --kill-on-invalid-dep=yes \
  -J ni26-d1-collect -o logs/d1-collect-%j.out \
  --wrap "set -e; $common; python -u laneC/full_collect.py '$root' '$work'")
printf '%s\n' "canary=$canary external=$extern full=$fit collector=$collector" | tee "$work/jobs.txt"
