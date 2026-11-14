#!/bin/bash
# s253 refits on Leonardo: one array task per order (20), its six bases in parallel (one thread each). From $WORK/ni26:
#   bash studies/14nov/causalpers/leo_array.sh ACCOUNT [HH:MM:SS]   -> runs/diag/cpu/cpers-<job>_<task>.out
set -euo pipefail
ACC=$1; TIME=${2:-02:00:00}; mkdir -p runs/diag/cpu runs/causalpers
sbatch --parsable --account=$ACC -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=$TIME --array=0-19 -J ni26c-cpers \
  -o runs/diag/cpu/cpers-%A_%a.out --wrap '
module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
O=(within0 within1 within2 within3 within4 within5 within6 within7 within8 within9 within10 within11 full0 full1 full2 full3 runs0 runs1 runs2 runs3)
o=${O[$SLURM_ARRAY_TASK_ID]}; date -u
for b in 0 1 2 3 4 5; do python -I studies/14nov/causalpers/run.py $o $b & done; wait
ls runs/causalpers/pp_${o}_b*.npz | wc -l; date -u'
