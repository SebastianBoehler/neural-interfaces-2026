#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 22
#SBATCH -J ni26-ext459
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
repo=$b/neural-interfaces-2026
first=$b/ni26_qualify385_v1_20261118
second=$b/ni26_cross411b_20261118
work=$PWD
sha256sum -c --status source.sha256
(cd "$b" && sha256sum -c --status "$work/external_parents.sha256")
(cd "$first" && sha256sum -c --status members.sha256 && sha256sum -c --status sources.sha256)
(cd "$b/ni26_kernel459_20261118/weights" && sha256sum -c --status "$work/models.sha256")
id=${SLURM_ARRAY_TASK_ID:-0}
person=$((id%9)); pair=$((id/9))
args=()
if [[ $1 == smoke ]]; then
    args=(smoke)
    person=5
    (cd "$first" && sha256sum -c --status inputs.sha256)
    echo "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed  $first/v66_cp1_cap250.zip" | sha256sum -c -
    (cd "$b/ni26_s318_coral_fbl_20261117" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$repo/.venv/bin/python" replay/check.py) > replay.log
fi
grep -Fx OK replay.log
archive=$first
if ((pair==1)); then archive=$second; fi
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
  --bind "$work:/work,$first/cap:/package:ro,$archive:/archive:ro,$b/ni26_kernel459_20261118/weights:/weights:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python external_worker.py "$person" "$pair" "${args[@]}"
