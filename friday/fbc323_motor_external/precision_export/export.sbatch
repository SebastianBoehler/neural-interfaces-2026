#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 5
#SBATCH -J ni26-f32e-export
#SBATCH -o export-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
work=${SLURM_SUBMIT_DIR:?}
origin=$base/ni26_s323_normsplit_20261117
f14=$base/ni26_friday_s1c_training_20261118
f16=$base/ni26_friday_s1c_distill_20261118
f32=$base/ni26_friday_s1c_motor_joint_20261119
image=$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif
cd "$work"
sha256sum --quiet -c actor.sha256
sha256sum --quiet -c export.sha256
(cd "$origin" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
(cd anchors_f16e && sha256sum --quiet -c actor.sha256)
printf '%s\n' 'f8d074e6867ebfda82383c164c257d64f154b251b0ccdbcb5153b297e7917723  actor.sha256' | \
    (cd "$f32" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
printf '%s  %s\n' 7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0 "$image" | sha256sum --quiet -c -
case "${1:?expected smoke or matrix}" in
    smoke) args=(0 --smoke) ;;
    matrix) args=("${SLURM_ARRAY_TASK_ID:?}") ;;
    *) exit 2 ;;
esac
pypath=$origin:$origin/experiments:$f14:$f16:$work/anchors_f16e:$f32
mounts=$work:$work:rw,$origin:$origin:ro,$f14:$f14:ro,$f16:$f16:ro,$f32:$f32:ro
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env "PYTHONPATH=$pypath,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" \
    --bind "$mounts" \
    --pwd "$work" "$image" python -B export_native.py "${args[@]}"
