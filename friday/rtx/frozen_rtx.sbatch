#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 16
#SBATCH -J ni26-frozen-rtx
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
work=$PWD
sha256sum --quiet -c rtx_source.sha256
(cd "$prior" && sha256sum --quiet -c "$work/rtx_parents.sha256" && sha256sum --quiet -c members.sha256)
args=()
if [[ $1 == smoke ]]; then
    args=(smoke)
    person=5
    (cd "$prior" && sha256sum --quiet -c inputs.sha256)
    [[ "$(git -C "$repo/2026-competition" rev-parse HEAD)" == a66bc23d1a0e686b895c093a642b45c74970758b ]]
    (cd "$base/ni26_s318_coral_fbl_20261117" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1         "$repo/.venv/bin/python" replay/check.py) > replay.log
else
    [[ $1 == pair ]]
    people=(0 5)
    person=${people[SLURM_ARRAY_TASK_ID]}
fi
grep -Fx OK replay.log
arms=(old cap)
if [[ $person == 5 ]]; then arms=(cap old); fi
for arm in "${arms[@]}"; do
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
      --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/prior \
      --bind "$work:/work:rw,$prior:/prior:ro,$prior/$arm:/package:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
      --pwd /work "$repo/runs/cloud/ni26_worker.sif" python frozen_rtx_worker.py "$arm" "$person" "${args[@]}"
done
(cd "$prior" && sha256sum --quiet -c members.sha256)
