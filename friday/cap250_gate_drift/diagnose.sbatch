#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:11:00
#SBATCH --job-name=ni26-c21gpackD
#SBATCH --output=c21gpackD-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
parent=$base/ni26_friday_c21gpack_live_v1_20261119
original=$base/ni26_friday_c21gpack_rtx_v1_20261119
known=$base/ni26_native366_exact_ea_normal_v1_20261117/reference_normal.json
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c source.sha256
(cd "$parent" && echo 'b0bc9a635389c4102107dabb9ceb7f13fc4f3a3d9f5ec2b3477160c0d267cc37  source.sha256' | sha256sum --quiet -c
 sha256sum --quiet -c source.sha256)
(cd "$original" && echo 'f9ef151c8934f6721b650cde5050ec0c99f1709696f1d1df385c57bd9ad990b7  source.sha256' | sha256sum --quiet -c
 sha256sum --quiet -c source.sha256)
case ${SLURM_ARRAY_TASK_ID:?} in
    0) mode=aa; fixture=recognized ;;
    1) mode=aa; fixture=unrecognized ;;
    2) mode=same; fixture=recognized ;;
    3) mode=same; fixture=unrecognized ;;
    *) exit 2 ;;
esac
mkdir -p "$work/.mpl-cache"
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/parent/friday/cap250_gate_package:/official,TMPDIR=/work,MPLCONFIGDIR=/work/.mpl-cache \
    --bind "$work:/work:rw,$parent:/parent:ro,$repo/2026-competition:/official:ro,$known:/known_predictions.json:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
    --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B diagnose.py "$mode" "$fixture" \
    /parent/v66_cp1_cap250.zip /parent/v66_cap250_gate_optimized.zip /input.npz /external.npz \
    /parent/known.json /parent/unknown.json /parent/reference_source.py /known_predictions.json
