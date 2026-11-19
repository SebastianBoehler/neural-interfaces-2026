#!/bin/bash
# Lane C2 (CBraMod): submit from the C2 release root. Needs data/ and runs/ (links to the Lane C release), laneC/, laneC2/.
set -euo pipefail
account=${1:?expected an authorized Leonardo Slurm account}
work=${2:?expected the absolute bank directory}
checkpoint=${3:?expected the absolute pinned CBraMod safetensors path}
venv=${4:?expected the absolute virtualenv path}
[[ "$work" = /* && "$checkpoint" = /* && "$venv" = /* && "$PWD" != *" "* && "$work" != *" "* ]]
test -s "$checkpoint"; test -s data/proteus_windows.npz; test -s runs/shuffle_perms.npz; test -d "$work"
mkdir -p logs
root=$PWD
sbatch_id() { sbatch --parsable "$@" | cut -d ';' -f1; }
common="module load python/3.11.7; unset PYTHONPATH; source $venv/bin/activate; export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
actor="python -u laneC2/train_cbramod.py"
args="--root '$root' --work '$work' --checkpoint '$checkpoint'"
canary=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t 00:25:00 \
  -J ni26-c2-canary -o logs/c2-canary-%j.out --wrap "set -e; $common; $actor canary 0 $args")
extern=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t 01:00:00 \
  --dependency=afterok:"$canary" --kill-on-invalid-dep=yes --array=0-2%3 \
  -J ni26-c2-external -o logs/c2-external-%A_%a.out \
  --wrap "set -e; $common; $actor external-full \$SLURM_ARRAY_TASK_ID $args")
fit=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t 01:00:00 \
  --dependency=afterok:"$extern" --kill-on-invalid-dep=yes --array=0-35%18 \
  -J ni26-c2-fit -o logs/c2-fit-%A_%a.out \
  --wrap "set -e; $common; $actor full \$SLURM_ARRAY_TASK_ID $args")
collector=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 2 --mem=16G -t 00:30:00 \
  --dependency=afterok:"$fit" --kill-on-invalid-dep=yes \
  -J ni26-c2-collect -o logs/c2-collect-%j.out \
  --wrap "set -e; $common; python -u laneC2/collect_cbramod.py '$root' '$work'")
printf '%s\n' "canary=$canary external=$extern full=$fit collector=$collector" | tee "$work/jobs.txt"
