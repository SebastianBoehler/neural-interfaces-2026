#!/bin/bash
# SBNX (11 Nov): smoke (1 epoch per architecture, tolerant) + 16-task array (8 architectures x seeds 0-1), 2080-galvani.
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
D=runs/diag/sbnx
mkdir -p $D/logs $D/probs $D/smoke
X=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
E="export PATH=\$PWD/.venv/bin:\$PATH OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONWARNINGS=ignore; \
python -c 'import torch; assert torch.cuda.is_available()' || exit 1; nvidia-smi --query-gpu=name --format=csv,noheader; hostname; set -e"
SMOKE=$(sbatch --parsable --account=gehler -p 2080-galvani --exclude=$X --gres=gpu:1 --cpus-per-task=3 --mem=32G \
  --time=00:40:00 --no-requeue -J ni26-sbnx-smoke -o $D/logs/smoke-%j.out --wrap "$E; python -I $D/galv/sbnx.py smoke")
ARR=$(sbatch --parsable --account=gehler -p 2080-galvani --exclude=$X --gres=gpu:1 --cpus-per-task=3 --mem=32G \
  --time=02:30:00 --no-requeue -J ni26-sbnx --array=0-15%8 -o $D/logs/sbnx-%A_%a.out \
  --wrap "$E; python -I $D/galv/sbnx.py \$SLURM_ARRAY_TASK_ID")
echo "$(date -u) smoke=$SMOKE array=$ARR" | tee -a $D/jobs.txt
