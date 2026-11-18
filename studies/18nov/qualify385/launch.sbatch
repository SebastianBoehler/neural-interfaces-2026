#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 12
#SBATCH -J ni26-qual385
set -eu
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
id=${SLURM_ARRAY_TASK_ID:--1}
sha256sum -c sources.sha256 >/dev/null
sha256sum -c members.sha256 >/dev/null
if [[ $id == -1 ]]; then
    sha256sum -c inputs.sha256
fi
arms=(old cap)
if (( id % 2 == 0 )); then arms=(cap old); fi
for arm in "${arms[@]}"; do
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
      --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
      --bind "$PWD:/work:rw,$PWD/$arm:/package:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro,$base/ni26_native366_exact_ea_normal_v1_20261117/reference_normal.json:/prior.json:ro" \
      --pwd /work "$repo/runs/cloud/ni26_worker.sif" python worker.py "$arm" "$id" >"$arm-$id.log" 2>&1
done
sha256sum -c members.sha256 >/dev/null
