#!/bin/bash
# Lane S (Shin 2017 / 2018 correctly mapped trunks for the raw FBCNet member). Run from the lane S release root.
# Usage: leo_lane_s.sh ACCOUNT WORKDIR VENV [smoke]
set -euo pipefail
acc=${1:?account}; work=${2:?bank dir}; venv=${3:?venv}; mode=${4:-full}
[[ "$work" = /* && "$venv" = /* && "$PWD" != *" "* ]]; test -s data/proteus_windows.npz; test -s runs/shuffle_perms.npz; mkdir -p logs "$work" runs/pretrain
root=$PWD
sb() { sbatch --parsable "$@" | cut -d ';' -f1; }
common="set -e; module load python/3.11.7; unset PYTHONPATH; source $venv/bin/activate; export LANEH_WORK=$work OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore; cd $root"
g="--account=$acc -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G"
if [ "$mode" = smoke ]; then
  p=$(sb $g -t 00:40:00 -J ni26-s-smoke-pt -o logs/s-smoke-pt-%j.out --wrap "$common; python -u experiments/test_class_map.py; python -u laneS/pretrain_s.py Z 0")
  f=$(sb $g -t 00:40:00 --dependency=afterok:$p --kill-on-invalid-dep=yes -J ni26-s-smoke-fit -o logs/s-smoke-fit-%j.out \
     --wrap "$common; python -u laneS/lopo.py Z 0 P02")
  echo "smoke pt=$p fit=$f"; exit 0
fi
pt=$(sb $g -t 06:00:00 --array=0-8 -J ni26-s-pt -o logs/s-pt-%A_%a.out \
   --wrap "$common; t=(P W D); python -u laneS/pretrain_s.py \${t[\$((SLURM_ARRAY_TASK_ID/3))]} \$((SLURM_ARRAY_TASK_ID%3))")
fit=$(sb $g -t 04:00:00 --dependency=afterok:$pt --kill-on-invalid-dep=yes --array=0-8 -J ni26-s-fit -o logs/s-fit-%A_%a.out \
   --wrap "$common; a=(P W D); python -u laneS/lopo.py \${a[\$((SLURM_ARRAY_TASK_ID/3))]} \$((SLURM_ARRAY_TASK_ID%3))")
echo "pt=$pt fit=$fit" | tee "$work/jobs.txt"
