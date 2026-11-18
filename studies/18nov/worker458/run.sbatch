#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 15
#SBATCH -J ni26-worker458
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
repo=$b/neural-interfaces-2026
ref=$b/ni26_external459_v2_20261118
arc=$b/ni26_qualify385_v1_20261118
native=$b/ni26_native458_20261118
weights=$b/ni26_kernel459_20261118/weights
work=$PWD
sha256sum --quiet -c source.sha256
(cd "$ref" && sha256sum --quiet -c source.sha256)
(cd "$weights" && sha256sum --quiet -c "$ref/models.sha256")
(cd "$arc" && sha256sum --quiet -c members.sha256)
(cd "$native" && echo 'eaac842f94a0acb8dca2815dedaf9abd00ef6865c8f9fae8e6d630feb05092c1  decoder.so' | sha256sum --quiet -c -)
(cd "$b" && sha256sum --quiet -c "$ref/external_parents.sha256")
case $1 in
 smoke) person=5
    rep=smoke
    arms=(graph)
    (cd "$b/ni26_s318_coral_fbl_20261117" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$repo/.venv/bin/python" replay/check.py) > replay.log ;;
 full) id=$SLURM_ARRAY_TASK_ID
    people=(0 5)
    person=${people[$((id%2))]}
    rep=$((id/2))
    arms=(stock graph)
    if (( (id%2+rep)%2 )); then arms=(graph stock); fi ;;
esac
grep -Fx OK replay.log
for arm in "${arms[@]}"; do
 apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs   --env PYTHONPATH=/reference,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl   --bind "$work:/work,$ref:/reference:ro,$arc/cap:/package:ro,$arc:/archive:ro,$native:/native:ro,$weights:/weights:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro"   --pwd /work "$repo/runs/cloud/ni26_worker.sif" python worker.py "$person" "$arm" "$rep"
done
