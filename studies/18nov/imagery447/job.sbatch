#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=13G
#SBATCH -t 12
#SBATCH -J ni26-chang447
set -eu
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
orig=$base/ni26_qualify385_v1_20261118
(cd "$orig" && sha256sum -c --status members.sha256 && sha256sum -c --status sources.sha256)
sha256sum -c --status sources.sha256
id=${SLURM_ARRAY_TASK_ID:--1}
arms=(old cap)
if (( id % 2 == 0 )); then arms=(cap old); fi
for arm in "${arms[@]}"; do
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
      --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/original \
      --bind "$PWD:/work:rw,$orig:/original:ro,$orig/$arm:/package:ro,$repo/2026-competition:/official:ro,$repo/data:/datasets:ro,$repo/data/proteus_windows.npz:/input.npz:ro,$base/ni26_native366_exact_ea_normal_v1_20261117/reference_normal.json:/prior.json:ro" \
      --pwd /work "$repo/runs/cloud/ni26_worker.sif" python worker447.py "$arm" "$id" >"$arm-$id.log" 2>&1
done
