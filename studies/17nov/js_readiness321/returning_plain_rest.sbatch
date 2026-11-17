#!/bin/bash
#SBATCH --job-name=ni26-js321-returning-plain-rest
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
idx=${SLURM_ARRAY_TASK_ID:?}
if ((idx<9)); then cfg=shallow_ea; within=$idx; nseed=5
elif ((idx<18)); then cfg=shallow_raw; within=$((idx-9)); nseed=5
elif ((idx<21)); then cfg=fbcnet_ea_tf; within=$((idx-18)); nseed=2
else cfg=fbcnet_raw_tf; within=$((idx-21)); nseed=2; fi
if ((within<nseed-1)); then fold=P04_s23; seed=$((within+1)); else fold=P10_s23; seed=$((within-nseed+1)); fi
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python returning_plain.py "$cfg" "$fold" "$seed"
