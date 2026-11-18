#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=13G
#SBATCH -t 20
#SBATCH -J ni26-ft422
set -eu
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
orig=$base/ni26_qualify385_v1_20261118
long=$base/ni26_select415_20261118
short=$base/ni26_short415_20261118
(cd "$orig" && sha256sum -c --status members.sha256 && sha256sum -c --status sources.sha256)
(cd "$long" && sha256sum -c --status sources.sha256)
sha256sum -c --status sources.sha256
id=${SLURM_ARRAY_TASK_ID:-${1:--1}}
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/original:/long \
  --bind "$PWD:/work:rw,$orig:/original:ro,$orig/cap:/package:ro,$long:/long:ro,$short:/short:ro,$repo/2026-competition:/official:ro,$repo/data:/datasets:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python worker422.py "$id"
