#!/bin/bash
# One task per GPU as a Slurm array on Galvani (13 Nov). Tasks must skip finished outputs, so a preempted task is simply
# requeued (--requeue). PART defaults to the non-preemptable A100 partition plus the preemptable ones.
# Usage (on Galvani, repo root): [PART=..] [DEP=--dependency=..] scripts/cloud/galv_array.sh NAME HH:MM:SS "python -I x.py" "0-23"
# Logs: runs/diag/arr/NAME-<array job>_<task>.out
set -euo pipefail
NAME=$1; TIME=$2; CMD=$3; IDX=$4; mkdir -p runs/diag/arr
sbatch --parsable ${DEP:-} --account=gehler -p ${PART:-a100-galvani,a100-preemptable-galvani,2080-preemptable-galvani} --gres=gpu:1 --cpus-per-task=8 --mem=64G \
  --time=$TIME --requeue --array=$IDX -J ni26g-$NAME -o runs/diag/arr/$NAME-%A_%a.out --wrap "
export PATH=\$PWD/.venv/bin:\$PATH NI26_WS=\$PWD OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONWARNINGS=ignore
python -c 'import torch; assert torch.cuda.is_available()' || exit 1
$CMD \$SLURM_ARRAY_TASK_ID; rc=\$?; echo EXIT \$rc; exit \$rc"
