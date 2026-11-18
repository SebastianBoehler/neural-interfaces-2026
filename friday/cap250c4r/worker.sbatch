#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 20
#SBATCH -J ni26-c4r
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
c3=$base/ni26_friday_c3_v1_20261118
c4=$base/ni26_friday_c4_v1_20261118
work=$PWD
sha256sum -c --status source.sha256
echo "8dd08259cbc95e8042544e3c893d7c53a354b9a851da97590520e16fb90cb92a  $c3/source.sha256" | sha256sum -c --status
echo "be10f8ec3cb19aeb9faa94616343ab994685ac28198e4f63920274c0fe37f65d  $c4/source.sha256" | sha256sum -c --status
(cd "$c3" && sha256sum -c --status source.sha256)
(cd "$c4" && sha256sum -c --status source.sha256)
(cd "$prior" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status members.sha256)
opts=()
if [[ $1 == smoke ]]; then
    index=5
    opts=(smoke)
    (cd "$prior" && sha256sum -c --status inputs.sha256)
    (cd "$base/ni26_s318_coral_fbl_20261117" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$repo/.venv/bin/python" replay/check.py) > replay.log
else
    [[ $1 == produce ]]
    index=$SLURM_ARRAY_TASK_ID
fi
grep -Fx OK replay.log
arms=(stock primary)
for j in 0 1; do
    arm=${arms[$(((j+index)%2))]}
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
      --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/tmp,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/c3:/c4:/prior \
      --bind "$work:/work:rw,$c3:/c3:ro,$c4:/c4:ro,$prior:/prior:ro,$prior/cap:/package:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
      --pwd /work "$repo/runs/cloud/ni26_worker.sif" python replicate.py "$index" "$arm" "${opts[@]}"
done
