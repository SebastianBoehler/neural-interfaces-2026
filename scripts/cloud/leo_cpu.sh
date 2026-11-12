#!/bin/bash
# Run one CPU study (replay scoring, own-space studies) as a job on Leonardo's CPU partition (dcgp_usr_prod), so many
# scorings run in parallel instead of one at a time on the Mac. Usage (from $WORK/ni26):
#   scripts/cloud/leo_cpu.sh NAME ACCOUNT HH:MM:SS "python -I studies/12nov/x.py args"   -> runs/diag/cpu/NAME-<job>.out
set -euo pipefail
NAME=$1; ACC=$2; TIME=$3; CMD=$4; mkdir -p runs/diag/cpu
sbatch --parsable ${DEP:-} --account=$ACC -p dcgp_usr_prod --cpus-per-task=16 --mem=100G --time=$TIME -J ni26c-$NAME -o runs/diag/cpu/$NAME-%j.out --wrap "
module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate
export NI26_WS=\$PWD NI26_ROOT=\$PWD OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NJ=12 PYTHONWARNINGS=ignore
date -u; $CMD; echo EXIT \$?; date -u"
