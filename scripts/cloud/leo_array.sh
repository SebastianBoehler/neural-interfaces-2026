#!/bin/bash
# One task per A100 as a Slurm job array (fixes the packing bottleneck of leo_pack.sh: tasks sharing one GPU slot
# compete for its 8 cores). Usage (from $WORK/ni26): scripts/cloud/leo_array.sh NAME ACCOUNT HH:MM:SS "python -I x.py" "0-23"
# Logs: runs/diag/arr/NAME-<array job>_<task>.out
set -euo pipefail
NAME=$1; ACC=$2; TIME=$3; CMD=$4; IDX=$5; mkdir -p runs/diag/arr
sbatch --parsable ${DEP:-} --account=$ACC -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=$TIME --array=$IDX -J ni26a-$NAME -o runs/diag/arr/$NAME-%A_%a.out --wrap "
module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate
export NI26_WS=\$PWD OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONWARNINGS=ignore
$CMD \$SLURM_ARRAY_TASK_ID; rc=\$?; echo EXIT \$rc; exit \$rc"  # a failing task must fail its Slurm job (13 Nov: sinc tasks showed COMPLETED)
