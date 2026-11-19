#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 23
#SBATCH -J ni26-fc11
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
normal=$base/ni26_native366_exact_ea_normal_v1_20261117
work=$PWD
sha256sum -c --status source.sha256
(cd "$prior" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status members.sha256)
id=$SLURM_ARRAY_TASK_ID
opts=()
if [[ $1 == smoke ]]; then
    opts=(smoke)
    (cd "$prior" && sha256sum -c --status inputs.sha256)
    echo "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327  $repo/data/proteus_windows.npz" | sha256sum -c --status
    (cd "$base/ni26_s318_coral_fbl_20261117" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$repo/.venv/bin/python" replay/check.py) > "replay-$id.log"
    grep -Fx OK "replay-$id.log"
else
    [[ $1 == produce ]]
    sha256sum -c --status smoke.sha256
fi
arms=(stock geometry rotation)
for j in 0 1 2; do
    arm=${arms[$(((j+id)%3))]}
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
      --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/tmp,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/prior \
      --bind "$work:/work:rw,$prior:/prior:ro,$prior/cap:/package:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro,$normal/reference_normal.json:/normal.json:ro" \
      --pwd /work "$repo/runs/cloud/ni26_worker.sif" python actor.py "$id" "$arm" "${opts[@]}"
done
