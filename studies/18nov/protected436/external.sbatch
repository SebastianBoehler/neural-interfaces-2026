#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 12
#SBATCH -J protect436ext
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
id=${SLURM_ARRAY_TASK_ID:--1}
sha256sum --quiet -c source.sha256
(( $(squeue --me -h -r -t RUNNING | wc -l) <= 32 ))
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
 --bind "$PWD:/work:rw,$base:$base:ro,$prior:/prior:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
 --pwd /work "$repo/runs/cloud/ni26_worker.sif" python external.py "$id"
