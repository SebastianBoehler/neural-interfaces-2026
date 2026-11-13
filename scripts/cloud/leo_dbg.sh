#!/bin/bash
# Short GPU job on the debug QoS (priority 80, 30 min wall time): profiling and checks that must turn around fast.
# Usage (from $WORK/ni26): scripts/cloud/leo_dbg.sh NAME ACCOUNT "python ..."   -> runs/diag/dbg/NAME-<job>.out
set -euo pipefail
NAME=$1; ACC=$2; CMD=$3; mkdir -p runs/diag/dbg
sbatch --parsable --account=$ACC -p boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=00:30:00 -J ni26d-$NAME -o runs/diag/dbg/$NAME-%j.out --wrap "
unset PYTHONPATH; . .venv312/bin/activate; export NI26_WS=\$PWD BENCHOPT_DATA_HOME=\$PWD/data OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
nvidia-smi --query-gpu=name --format=csv,noheader; date -u; $CMD; rc=\$?; echo EXIT \$rc; date -u; exit \$rc"
