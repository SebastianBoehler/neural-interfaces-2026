#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --gres=none
#SBATCH --array=0-3%4
#SBATCH --job-name=ni26-s337-pretraining
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export PRETRAIN_ROOT="$PWD" TEACHER_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_pretrain337_targets_v1_20261117 SMOKE_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_pretrain337_smoke_v2_20261117 NI26_DEVICE=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TORCHINDUCTOR_CACHE_DIR="/tmp/ni26_inductor_${SLURM_JOB_ID}" TRITON_CACHE_DIR="/tmp/ni26_triton_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B producer.py "$SLURM_ARRAY_TASK_ID"
