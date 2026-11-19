#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 5
#SBATCH -J ni26-f25e-cpu
#SBATCH -o export-%A_%a.out
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
work=${SLURM_SUBMIT_DIR:?}
o=$b/ni26_s323_normsplit_20261117
f14=$b/ni26_friday_s1c_training_20261118
f16=$b/ni26_friday_s1c_distill_20261118
f17=$b/ni26_friday_s1c_phase_20261118
f25=$b/ni26_friday_s1c_modulation_20261119
d=$b/ni26_friday_s1c_motor_external_cpu_20261119
stream=$b/ni26_friday_return_c20_actors_v2_20261119
image=$b/neural-interfaces-2026/runs/cloud/ni26_worker.sif
cd "$work"
sha256sum --quiet -c actor.sha256
sha256sum --quiet -c export.sha256
(cd "$o" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
(cd "$f25" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
printf '%s\n' '0ff4ad0010558c481de6a4faaeda40e5678574dcff6e888d7b70ff798f68184a  export.sha256' | \
    (cd "$d" && sha256sum --quiet -c - && sha256sum --quiet -c export.sha256 && sha256sum --quiet -c actor.sha256)
printf '%s\n' '394419b7f24e7c17528618914aab8ad10f475de3c949c1f406ee053fecd95c91  incremental_model.py' | \
    (cd "$stream" && sha256sum --quiet -c -)
printf '%s  %s\n' 7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0 "$image" | sha256sum --quiet -c -
case "${1:?expected smoke, matrix or seal}" in
    smoke) args=(export_external.py 0 --smoke) ;;
    matrix) args=(export_external.py "${SLURM_ARRAY_TASK_ID:?}") ;;
    seal) args=(seal.py) ;;
    *) exit 2 ;;
esac
pypath=$o:$o/experiments:$f14:$f16:$f17:$f25:$d:$d/anchors_f16e:$stream
mounts=$work:$work:rw,$o:$o:ro,$f14:$f14:ro,$f16:$f16:ro,$f17:$f17:ro,$f25:$f25:ro,$d:$d:ro
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
    --env "PYTHONPATH=$pypath,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" \
    --bind "$mounts,$stream/incremental_model.py:$stream/incremental_model.py:ro" \
    --pwd "$work" "$image" python -B "${args[@]}"
