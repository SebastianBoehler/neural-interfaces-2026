#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 12
#SBATCH -J ni26-friday-f5
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
work=$PWD
sha256sum -c --status external_source.sha256
(cd "$prior" && sha256sum -c --status "$work/external_parents.sha256" && sha256sum -c --status members.sha256)
grep -Fx OK replay.log
args=()
if [[ $1 == smoke ]]; then
    args=(smoke)
    id=5
else
    [[ $1 == produce ]]
    id=${SLURM_ARRAY_TASK_ID:?}
    [[ -s external_smoke.json ]]
fi
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/prior \
  --bind "$work:/work,$prior:/prior:ro,$prior/cap:/package:ro,$work/weights:/weights:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python external_worker.py "$id" "${args[@]}"
