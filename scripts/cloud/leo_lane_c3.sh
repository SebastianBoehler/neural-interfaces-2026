#!/bin/bash
# Lane C3 (Pratibha-2): submit from the C3 release root (data/, runs/ links, laneC/, laneC3/, pratibha_<rev>/ checkpoint dir).
# mode=smoke submits the canary plus two stage-B-only arm-A diagnostics; mode=full submits the wave (external x3, 36 fits, collector).
set -euo pipefail
mode=${1:?smoke|full}; account=${2:?account}; work=${3:?bank dir}; checkpoint=${4:?pinned model.safetensors}; venv=${5:?venv}
[[ "$work" = /* && "$checkpoint" = /* && "$venv" = /* && "$PWD" != *" "* ]]
test -s "$checkpoint"; test -s data/proteus_windows.npz; test -s runs/shuffle_perms.npz; mkdir -p "$work" logs
root=$PWD
sbatch_id() { sbatch --parsable "$@" | cut -d ';' -f1; }
common="module load python/3.11.7; unset PYTHONPATH; source $venv/bin/activate; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
actor="python -u laneC3/train_pratibha.py"
args="--root '$root' --work '$work' --checkpoint '$checkpoint'"
gpu="--account=$account -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G"
if [ "$mode" = smoke ]; then
  s=$(sbatch_id $gpu -t 00:10:00 -J ni26-c3-smoke -o logs/c3-smoke-%j.out --wrap "set -e; $common; python -u laneC3/smoke_pratibha.py '$root' '$checkpoint'")
  c=$(sbatch_id $gpu -t 00:25:00 --dependency=afterok:"$s" --kill-on-invalid-dep=yes -J ni26-c3-canary -o logs/c3-canary-%j.out --wrap "set -e; $common; $actor canary 0 $args")
  mkdir -p "$work/diagA"
  d=$(sbatch_id $gpu -t 01:00:00 --dependency=afterok:"$c" --kill-on-invalid-dep=yes --array=0,3 -J ni26-c3-diagA -o logs/c3-diagA-%A_%a.out \
    --wrap "set -e; $common; $actor proteus \$SLURM_ARRAY_TASK_ID --root '$root' --work '$work/diagA' --checkpoint '$checkpoint'")
  echo "smoke=$s canary=$c diagA=$d" | tee -a "$work/jobs.txt"; exit 0
fi
extern=$(sbatch_id $gpu -t 01:00:00 --array=0-2%3 -J ni26-c3-external -o logs/c3-external-%A_%a.out \
  --wrap "set -e; $common; $actor external-full \$SLURM_ARRAY_TASK_ID $args")
fit=$(sbatch_id $gpu -t 01:30:00 --dependency=afterok:"$extern" --kill-on-invalid-dep=yes --array=0-35%18 -J ni26-c3-fit -o logs/c3-fit-%A_%a.out \
  --wrap "set -e; $common; $actor full \$SLURM_ARRAY_TASK_ID $args")
col=$(sbatch_id --account="$account" -p boost_usr_prod --gres=gpu:1 -c 2 --mem=16G -t 00:30:00 --dependency=afterok:"$fit" --kill-on-invalid-dep=yes \
  -J ni26-c3-collect -o logs/c3-collect-%j.out --wrap "set -e; $common; python -u laneC3/collect_pratibha.py '$root' '$work'")
echo "external=$extern full=$fit collector=$col" | tee -a "$work/jobs.txt"
