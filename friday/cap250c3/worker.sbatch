#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 22
#SBATCH -J ni26-friday-c3
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
work=$PWD
sha256sum -c --status source.sha256
(cd "$prior" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status members.sha256)
args=()
if [[ $1 == smoke ]]; then
    args=(smoke)
    id=5
    (cd "$prior" && sha256sum -c --status inputs.sha256)
    echo "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed  $prior/v66_cp1_cap250.zip" | sha256sum -c --status
    (cd "$base/ni26_s318_coral_fbl_20261117" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$repo/.venv/bin/python" replay/check.py) > replay.log
else
    [[ $1 == produce ]]
    id=$SLURM_ARRAY_TASK_ID
fi
grep -Fx OK replay.log
arms=(stock primary placebo)
for n in 0 1 2; do
    arm=${arms[$(((n + id) % 3))]}
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
      --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/prior \
      --bind "$work:/work:rw,$prior:/prior:ro,$prior/cap:/package:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
      --pwd /work "$repo/runs/cloud/ni26_worker.sif" python external_actor.py "$id" "$arm" "${args[@]}"
done
