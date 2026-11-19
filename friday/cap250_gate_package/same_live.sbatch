#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:12:00
#SBATCH --job-name=ni26-c21gpackL
#SBATCH --output=c21gpackL-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
stock="$work/v66_cp1_cap250.zip"
mkdir -p "$work/.mpl-cache"
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/official,TMPDIR=/work,MPLCONFIGDIR=/work/.mpl-cache \
    --bind "$work:/work:rw,$repo/2026-competition:/official:ro,$stock:/stock.zip:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
    --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B friday/cap250_gate_package/same_live.py \
    /stock.zip /work/v66_cap250_gate_optimized.zip /input.npz /external.npz /work/known.json /work/unknown.json /work/reference_source.py
