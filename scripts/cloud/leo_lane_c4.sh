#!/bin/bash
# Lane C4 (frozen BrainCodes / NeuroRVQ + head): submit from the C4 release root (data/, runs/ links, laneC/, laneC4/, braincodes/, biocodec/, neurorvq/, bd_src/).
# mode=diag submits two stage-B-only arm-A fits; mode=smoke submits the two-model smoke plus the one-step canary for MODEL/INIT; mode=full submits external x3, 36 fits and the collector for MODEL/INIT.
set -euo pipefail
mode=${1:?smoke|full}; account=${2:?account}; model=${3:?braincodes|neurorvq}; init=${4:?pretrained|random}; venv=${5:?venv}; input=${6:-aligned}
[[ "$account" =~ ^[A-Za-z0-9_]+$ && "$venv" = /* && "$PWD" != *" "* ]]
case "$model" in braincodes) checkpoint=$PWD/braincodes/braincodes.pt;; neurorvq) checkpoint=$PWD/neurorvq/model.safetensors;; *) exit 2;; esac
[[ "$init" = pretrained || "$init" = random ]]; [[ "$input" = aligned || "$input" = raw ]]
test -s "$checkpoint"; test -s data/proteus_windows.npz; test -s runs/shuffle_perms.npz
root=$PWD; work=$root/bank_${model}_${init}_f3_${input}; mkdir -p "$work" logs
sbatch_id() { sbatch --parsable "$@" | cut -d ';' -f1; }
common="module load python/3.11.7; unset PYTHONPATH; source $venv/bin/activate; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
actor="python -u laneC4/train_c4.py"
args="--root '$root' --work '$work' --checkpoint '$checkpoint' --model $model --init $init --input $input"
gpu="--account=$account -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G"
if [ "$mode" = diag ]; then  # stage-B-only (arm A) fits of P02 and P10 for the training-loss trainability check; held recall is not used
  mkdir -p "$work/diagA"
  d=$(sbatch_id $gpu -t 01:00:00 --array=0,3 -J ni26-c4-diagA-$model-$init-$input -o logs/c4-diagA-$model-$init-$input-%A_%a.out \
    --wrap "set -e; $common; $actor proteus \$SLURM_ARRAY_TASK_ID --root '$root' --work '$work/diagA' --checkpoint '$checkpoint' --model $model --init $init --input $input")
  echo "diagA=$d" | tee -a "$work/jobs.txt"; exit 0
fi
if [ "$mode" = smoke ]; then
  s=$(sbatch_id $gpu -t 00:20:00 -J ni26-c4-smoke -o logs/c4-smoke-%j.out --wrap "set -e; $common; python -u laneC4/smoke_c4.py '$root' '$root'")
  c=$(sbatch_id $gpu -t 00:25:00 --dependency=afterok:"$s" --kill-on-invalid-dep=yes -J ni26-c4-canary-$model-$init-$input -o logs/c4-canary-$model-$init-$input-%j.out \
    --wrap "set -e; $common; $actor canary 0 $args")
  echo "smoke=$s canary=$c" | tee -a "$work/jobs.txt"; exit 0
fi
extern=$(sbatch_id $gpu -t 01:00:00 --array=0-2%3 -J ni26-c4-ext-$model-$init-$input -o logs/c4-external-$model-$init-$input-%A_%a.out \
  --wrap "set -e; $common; $actor external-full \$SLURM_ARRAY_TASK_ID $args")
fit=$(sbatch_id $gpu -t 01:30:00 --dependency=afterok:"$extern" --kill-on-invalid-dep=yes --array=0-35%18 -J ni26-c4-fit-$model-$init-$input -o logs/c4-fit-$model-$init-$input-%A_%a.out \
  --wrap "set -e; $common; $actor full \$SLURM_ARRAY_TASK_ID $args")
col=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 2 --mem=16G -t 00:30:00 --dependency=afterok:"$fit" --kill-on-invalid-dep=yes \
  -J ni26-c4-collect-$model-$init-$input -o logs/c4-collect-$model-$init-$input-%j.out --wrap "set -e; $common; python -u laneC4/collect_c4.py '$root' '$work' $model-$init $input")
echo "external=$extern full=$fit collector=$col" | tee -a "$work/jobs.txt"
