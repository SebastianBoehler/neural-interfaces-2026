#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 10
#SBATCH -J ni26-friday-c8
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
c6=$base/ni26_friday_c6_v2_20261118
c7=$base/ni26_friday_c7_v1_20261118
work=$PWD
sha256sum -c --status source.sha256
echo "6ec797665c15d8218b758af8f1a53f895afd3d58822f2dc7a00f5258a88f7ced  $c6/source.sha256" | sha256sum -c --status
echo "7927abf1d34bbe8b35b5d5a03dd30dd64e4f33461cd44341ed5bd0714baf4e22  $c7/source.sha256" | sha256sum -c --status
(cd "$c6" && sha256sum -c --status source.sha256)
(cd "$c7" && sha256sum -c --status source.sha256)
(cd "$prior" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status members.sha256)
args=()
if [[ $1 == smoke ]]; then
    args=(smoke)
    id=5
    (cd "$prior" && sha256sum -c --status inputs.sha256)
    (cd "$base/ni26_s318_coral_fbl_20261117" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$repo/.venv/bin/python" replay/check.py) > replay.log
else
    [[ $1 == produce ]]
    id=$SLURM_ARRAY_TASK_ID
fi
grep -Fx OK replay.log
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/c7:/c6:/prior \
    --bind "$work:/work:rw,$prior:/prior:ro,$prior/cap:/package:ro,$c6:/c6:ro,$c7:/c7:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
    --pwd /work "$repo/runs/cloud/ni26_worker.sif" python replicate.py "$id" "${args[@]}"
