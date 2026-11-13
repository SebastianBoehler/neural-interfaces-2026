#!/bin/bash
# Run one CPU study (replay scoring, own-space studies) as a job on a Leonardo booster slot (our budgets are GPU-only; dcgp refuses them), so many
# scorings run in parallel instead of one at a time on the Mac. Usage (from $WORK/ni26):
#   scripts/cloud/leo_cpu.sh NAME ACCOUNT HH:MM:SS "python -I studies/12nov/x.py args"   -> runs/diag/cpu/NAME-<job>.out
set -euo pipefail
NAME=$1; ACC=$2; TIME=$3; CMD=$4; mkdir -p runs/diag/cpu
sbatch --parsable ${DEP:-} --account=$ACC -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=120G --time=$TIME -J ni26c-$NAME -o runs/diag/cpu/$NAME-%j.out --wrap "
module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate
export NI26_WS=\$PWD NI26_ROOT=\$PWD OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NJ=6 PYTHONWARNINGS=ignore
date -u; $CMD; rc=\$?; echo EXIT \$rc; date -u; exit \$rc"
