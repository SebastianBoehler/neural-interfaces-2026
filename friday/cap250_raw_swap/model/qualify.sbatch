#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 5
#SBATCH -J ni26-fc13-model
#SBATCH -o model-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
origin=$base/ni26_s323_normsplit_20261117
f14=$base/ni26_friday_s1c_training_20261118
f15=$base/ni26_friday_s1c_skip_20261118
f15e=$base/ni26_friday_s1c_skip_external_20261118
f16=$base/ni26_friday_s1c_distill_20261118
f16e=$base/ni26_friday_s1c_distill_external_20261118
cross=$base/ni26_cross411b_20261118
work=${SLURM_SUBMIT_DIR:?}
image=$repo/runs/cloud/ni26_worker.sif
cd "$work"
sha256sum --quiet -c source.sha256
(cd "$origin" && sha256sum --quiet -c source323.sha256)
(cd "$f15e" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c score.sha256 && sha256sum --quiet -c actor.sha256)
(cd "$f16e" && sha256sum --quiet -c score.sha256)
(cd "$cross" && sha256sum --quiet -c "$f15e/orders.sha256")
if [[ "${1:?export or complete}" == export ]]; then
    person=5
    printf '%s  %s\n' 7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0 "$image" | sha256sum --quiet -c -
else
    person=${SLURM_ARRAY_TASK_ID:?}
fi
pypath=$origin:$origin/experiments:$f14:$f15:$f15e:$f16:$f16e:$cross
mounts=$work:$work:rw,$origin:$origin:ro,$f14:$f14:ro,$f15:$f15:ro,$f15e:$f15e:ro,$f16:$f16:ro,$f16e:$f16e:ro,$cross:$cross:ro
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env "PYTHONPATH=$pypath,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" \
    --bind "$mounts,$repo:$repo:ro" --pwd "$work" "$image" python qualify_model.py "$1" "$person"
