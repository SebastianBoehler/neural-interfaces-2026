#!/bin/bash
# Lane V on Leonardo (Ferranti H100 queue was ~2 days deep). Run from the release root: leo_fit.sh smoke | run
set -euo pipefail
root=$PWD; mode=${1:?}; acc=AIFAC_P02_1675
common="set -e; module load python/3.11.7; unset PYTHONPATH; source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate; export LANEH_WORK=$root/work OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1; cd $root"
g="--account=$acc -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G"
mkdir -p logs work work_smoke
if [ "$mode" = smoke ]; then
  sbatch --parsable $g -t 00:25:00 -J ni26-v-smoke -o logs/v-smoke-%j.out --wrap "$common; export LANEH_WORK=$root/work_smoke LANEH_QUICK=1; for a in V0 V1 V2 V2s V3; do python -u laneV/lopo_v.py \$a 0 P02; done"
else
  sbatch --parsable $g -t 01:00:00 --array=0-14 -J ni26-v-fit -o logs/v-fit-%A_%a.out --wrap "$common; a=(V0 V1 V2 V2s V3); python -u laneV/lopo_v.py \${a[\$((SLURM_ARRAY_TASK_ID/3))]} \$((SLURM_ARRAY_TASK_ID%3))"
fi
