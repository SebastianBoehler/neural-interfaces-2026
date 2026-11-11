#!/bin/bash
# Pack many small training tasks into ONE Slurm job: P tasks run concurrently on one GPU, and GPU utilisation is logged.
# Why: one task (12 folds of a small net) takes 3-6 min of a 2080 Ti but waited 10-60+ min in the queue on 11 Nov; an
# array of 21 tasks ran 1-2 at a time. A packed job waits once and uses the GPU the single tasks leave idle.
# Usage (on Galvani, from the repo root):
#   scripts/cloud/pack.sh NAME P HH:MM:SS "python -I runs/diag/w12/galv/w12.py" 11 12 13 ...
# Logs: runs/diag/pack/NAME-<job>-task<T>.out, runs/diag/pack/NAME-<job>-gpu.csv (utilisation every 20 s).
set -euo pipefail
NAME=$1; P=$2; TIME=$3; CMD=$4; shift 4
D=runs/diag/pack; mkdir -p $D
X=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
sbatch --parsable --account=gehler -p 2080-galvani --exclude=$X --gres=gpu:1 --cpus-per-task=8 --mem=64G \
  --time=$TIME --no-requeue -J ni26-$NAME -o $D/$NAME-%j.out --wrap "
export PATH=\$PWD/.venv/bin:\$PATH OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONWARNINGS=ignore
python -c 'import torch; assert torch.cuda.is_available()' || exit 1
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date -u
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 20 > $D/$NAME-\$SLURM_JOB_ID-gpu.csv &
SMI=\$!
printf '%s\n' $* | xargs -P $P -I{} sh -c '$CMD {} > $D/$NAME-'\$SLURM_JOB_ID'-task{}.out 2>&1; echo task {} exit \$? \$(date -u +%H:%M:%S)'
kill \$SMI; date -u; awk -F', ' '{gsub(/ %/,\"\",\$2); s+=\$2; n++} END {print \"mean GPU utilisation %\", s/n, \"samples\", n}' $D/$NAME-\$SLURM_JOB_ID-gpu.csv"
