#!/bin/bash
# Run one CPU study (replay scoring) as a job on Galvani's cpu partition, so scoring chains on the cluster and does not
# depend on the SSH master staying up (13 Nov, ledger s202). Usage (on Galvani, repo root):
#   [DEP=--dependency=..] scripts/cloud/galv_cpu.sh NAME HH:MM:SS "python -I studies/13nov/x.py args"   -> runs/diag/cpu/NAME-<job>.out
set -euo pipefail
NAME=$1; TIME=$2; CMD=$3; mkdir -p runs/diag/cpu
sbatch --parsable ${DEP:-} --account=gehler -p ${PART:-cpu-galvani} --cpus-per-task=8 --mem=96G --time=$TIME -J ni26c-$NAME -o runs/diag/cpu/$NAME-%j.out --wrap "
export PATH=\$PWD/.venv/bin:\$PATH NI26_WS=\$PWD NI26_ROOT=\$PWD OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NJ=6 PYTHONWARNINGS=ignore
date -u; $CMD; rc=\$?; echo EXIT \$rc; date -u; exit \$rc"
