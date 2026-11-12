#!/bin/bash
# Leonardo (CINECA) version of pack2.sh: P tasks run concurrently on one A100 of the booster partition.
# Usage (on Leonardo, from $WORK/ni26): scripts/cloud/leo_pack.sh NAME ACCOUNT P HH:MM:SS "python -I script.py" T1 T2 ...
# Logs: runs/diag/pack/NAME-<job>-task<T>.out. NI26_WS points the study scripts at this folder.
set -euo pipefail
NAME=$1; ACC=$2; P=$3; TIME=$4; CMD=$5; shift 5
D=runs/diag/pack; mkdir -p $D
sbatch --parsable ${DEP:-} --account=$ACC -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=120G --time=$TIME -J ni26-$NAME -o $D/$NAME-%j.out --wrap "
module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate
export NI26_WS=\$PWD OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore
python -c 'import torch; assert torch.cuda.is_available()' || exit 1
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date -u
printf '%s\n' $* | xargs -P $P -I{} sh -c '$CMD {} > $D/$NAME-'\$SLURM_JOB_ID'-task{}.out 2>&1; echo task {} exit \$? \$(date -u +%H:%M:%S)'
date -u"
