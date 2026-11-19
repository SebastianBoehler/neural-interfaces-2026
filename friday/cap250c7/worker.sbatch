#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 10
#SBATCH -J ni26-friday-c7-worker
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
external=$base/ni26_qualify385_v1_20261118
c6=$base/ni26_friday_c6_v2_20261118
work=$PWD
sha256sum -c --status source.sha256
sha256sum -c --status coefficients.sha256
echo "6ec797665c15d8218b758af8f1a53f895afd3d58822f2dc7a00f5258a88f7ced  $c6/source.sha256" | sha256sum -c --status
(cd "$c6" && sha256sum -c --status source.sha256)
(cd "$external" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status members.sha256)
grep -Fx OK replay.log
args=()
if [[ $1 == smoke ]]; then
    args=(smoke)
    id=5
    (cd "$external" && sha256sum -c --status inputs.sha256)
    echo "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed  $external/v66_cp1_cap250.zip" | sha256sum -c --status
else
    [[ $1 == produce ]]
    id=$SLURM_ARRAY_TASK_ID
fi
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/c6:/prior \
  --bind "$work:/work:rw,$external:/prior:ro,$external/cap:/package:ro,$c6:/c6:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python actor.py "$id" "${args[@]}"
