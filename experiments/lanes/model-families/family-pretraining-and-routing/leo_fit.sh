#!/bin/bash
# Lane B4 on Leonardo. Run from the release root: leo_fit.sh smoke | fit | time
# fit: array task = family_index*12 + seed (raw, eafbc, fblight seeds 0-11; ids 36-39 = sinc seeds 0-3, the only m7r trunks)
set -euo pipefail
root=$PWD; mode=${1:?}; acc=AIFAC_P02_1675
common="set -e; module load python/3.11.7; unset PYTHONPATH; source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate; export LANEH_WORK=$root/work OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1; cd $root"
g="--account=$acc -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G"
mkdir -p logs work work_smoke
fams="(raw eafbc fblight sinc)"
if [ "$mode" = smoke ]; then  # code path only: 1 epoch, one fold, every family
  sbatch --parsable $g -t 00:25:00 -J ni26-b4-smoke -o logs/b4-smoke-%j.out --wrap "$common; export LANEH_WORK=$root/work_smoke LANEH_QUICK=1; for f in raw eafbc fblight sinc; do python -u laneB4/lopo_b4.py \$f 0 P02; done"
elif [ "$mode" = fit ]; then
  sbatch --parsable $g -t 01:00:00 --array=${2:-0-39} -J ni26-b4-fit -o logs/b4-fit-%A_%a.out --wrap "$common; f=$fams; id=\$SLURM_ARRAY_TASK_ID; fam=\${f[\$((id/12))]}; python -u laneB4/lopo_b4.py \$fam \$((id%12))"
fi
