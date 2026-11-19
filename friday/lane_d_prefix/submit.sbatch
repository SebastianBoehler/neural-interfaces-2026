#!/bin/bash
#SBATCH --job-name=ni26-lane-d
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --time=00:50:00
#SBATCH --mem=12G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
repo=${LANE_D_REPO:-/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026}
image=${LANE_D_IMAGE:-$repo/runs/cloud/ni26_worker.sif}
mode=${1:?}
sha256sum --quiet -c source.sha256
bind="$PWD:/work,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro"
case "$mode" in
smoke) item=40 ;;
run) item=${SLURM_ARRAY_TASK_ID:?}; grep -Fx PASS replay.ok ;;
collect)
    exec apptainer exec --cleanenv --bind "$bind" --pwd /work "$image" \
        python collect.py /work /external.npz
    ;;
*) exit 2 ;;
esac
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work \
    --bind "$bind" --pwd /work "$image" \
    python worker.py /work /external.npz "$item" "$mode" cuda:0
if [ "$mode" = smoke ]; then
    printf '%s\n' PASS > replay.ok
fi
