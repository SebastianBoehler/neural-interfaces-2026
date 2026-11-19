#!/bin/bash
# Lane H (pooled-corpus base for the raw FBCNet member). Run from the lane H release root (needs experiments/, laneH/,
# data/, runs/, 2026-competition/). Usage: leo_lane_h.sh ACCOUNT WORKDIR VENV [smoke]
set -euo pipefail
acc=${1:?account}; work=${2:?bank dir}; venv=${3:?venv}; mode=${4:-full}
[[ "$work" = /* && "$venv" = /* && "$PWD" != *" "* ]]; test -s data/proteus_windows.npz; test -s runs/shuffle_perms.npz; mkdir -p logs "$work" runs/pretrain
root=$PWD
sb() { sbatch --parsable "$@" | cut -d ';' -f1; }
common="set -e; module load python/3.11.7; unset PYTHONPATH; source $venv/bin/activate; export LANEH_WORK=$work OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore; cd $root"
g="--account=$acc -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G"
if [ "$mode" = smoke ]; then
  p=$(sb $g -t 00:40:00 -J ni26-h-smoke-pt -o logs/h-smoke-pt-%j.out --wrap "$common; python -u laneH/pretrain_h.py hs9 0")
  f=$(sb $g -t 00:40:00 --dependency=afterok:$p --kill-on-invalid-dep=yes -J ni26-h-smoke-fit -o logs/h-smoke-fit-%j.out \
     --wrap "$common; python -u laneH/lopo.py S 0 P02")
  a=$(sb $g -t 00:40:00 -J ni26-h-smoke-a -o logs/h-smoke-a-%j.out --wrap "$common; LANEH_QUICK=1 python -u laneH/lopo.py A 0 P02")
  echo "smoke pt=$p fit=$f a=$a"; exit 0
fi
pt=$(sb $g -t 06:00:00 --array=0-5 -J ni26-h-pt -o logs/h-pt-%A_%a.out \
   --wrap "$common; t=(hp7 hp9); python -u laneH/pretrain_h.py \${t[\$((SLURM_ARRAY_TASK_ID/3))]} \$((SLURM_ARRAY_TASK_ID%3))")
# arm A needs no trunk: starts at once; B (tasks 3-5) and C (6-8) wait for their own seed's trunk
fitA=$(sb $g -t 04:00:00 --array=0-2 -J ni26-h-fitA -o logs/h-fitA-%A_%a.out --wrap "$common; python -u laneH/lopo.py A \$SLURM_ARRAY_TASK_ID")
fitB=$(sb $g -t 04:00:00 --dependency=afterok:$pt --kill-on-invalid-dep=yes --array=0-5 -J ni26-h-fitBC -o logs/h-fitBC-%A_%a.out \
   --wrap "$common; a=(B C); python -u laneH/lopo.py \${a[\$((SLURM_ARRAY_TASK_ID/3))]} \$((SLURM_ARRAY_TASK_ID%3))")
echo "pt=$pt fitA=$fitA fitBC=$fitB" | tee "$work/jobs.txt"
