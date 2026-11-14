#!/bin/bash
# s253 selection-form refits on Galvani CPU: one array task per order (20), six bases in parallel (one thread each).
# From the Galvani repo root: [DRAW=d1|d2] [PIPES=a2] [PART=cpu-galvani] [FORMS=...] bash studies/14nov/causalpers/galv_array.sh [HH:MM:SS]
set -euo pipefail
TIME=${1:-02:00:00}; F=${FORMS:-top,cq50,cq25,agree,agcq}; mkdir -p runs/diag/cpu runs/causalpers
sbatch --parsable --account=gehler -p ${PART:-cpu-galvani} --cpus-per-task=8 --mem=48G --time=$TIME --array=0-19 -J ni26c-cpers \
  -o runs/diag/cpu/cpers-%A_%a.out --wrap "
export PATH=\$PWD/.venv/bin:\$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore FORMS=$F DRAW=${DRAW:-} PIPES=${PIPES:-l1,a2,a4,lE}
O=(within0 within1 within2 within3 within4 within5 within6 within7 within8 within9 within10 within11 full0 full1 full2 full3 runs0 runs1 runs2 runs3)
o=\${O[\$SLURM_ARRAY_TASK_ID]}; date -u
for b in 0 1 2 3 4 5; do python -I studies/14nov/causalpers/run.py \$o \$b & done; wait; date -u"
