#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 22
#SBATCH -J ni26-step448
set -eu
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
id=${SLURM_ARRAY_TASK_ID:--1}
sha256sum -c sources.sha256 --status
(cd "$prior" && sha256sum -c members.sha256 --status)
if [[ $id == -1 ]]; then
    (cd "$prior" && sha256sum -c inputs.sha256 --status && sha256sum -c sources.sha256 --status)
fi
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/prior \
  --bind "$PWD:/work:rw,$prior:/prior:ro,$prior/cap:/package:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python worker.py "$id"
