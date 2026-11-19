#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:06:00
#SBATCH --job-name=ni26-c20r-actors
#SBATCH --output=c20r-actor-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
origin=$base/ni26_s323_normsplit_20261117
f14=$base/ni26_friday_s1c_training_20261118
f16=$base/ni26_friday_s1c_distill_20261118
f17=$base/ni26_friday_s1c_phase_20261118
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c source.sha256
(cd "$origin" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c score.sha256 && sha256sum --quiet -c actor.sha256)
(cd "$f17" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256 && sha256sum --quiet -c before_labels.sha256)
image=$repo/runs/cloud/ni26_worker.sif
printf '%s  %s\n' 7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0 "$image" | sha256sum --quiet -c -
people=(P04 P10)
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env "PYTHONPATH=$origin:$origin/experiments:$f14:$f16:$f17,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1" \
  --bind "$base:$base:ro,$work:$work:rw" --pwd "$work" "$image" python -B export_pupils.py "${people[${SLURM_ARRAY_TASK_ID:?}]}"
