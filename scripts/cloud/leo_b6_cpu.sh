#!/bin/bash
# Lane B6 serial/internet job (lrd_all_serial): leo_b6_cpu.sh SCRIPT ARGS...  Env R_ROOT = release root (set by sbatch --export).
set -e; cd "${R_ROOT:?}"
module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH
export MNE_DATA=$PWD/data/mi84_raw PYTHONWARNINGS=ignore OMP_NUM_THREADS=2
exec /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/python -I "$@"
