#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --gres=none
#SBATCH --array=0-1%2
#SBATCH --job-name=ni26-s345-taskdistill
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export DISTILL_ROOT="$PWD" NI26_DEVICE=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export SMOKE_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_distill345_smoke_v3_20261117 TEACHER_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_distill345_teacher_v3_20261117
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B run345.py "$SLURM_ARRAY_TASK_ID"
