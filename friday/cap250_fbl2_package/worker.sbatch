#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-c21p
#SBATCH --output=c21p-%j.out
set -euo pipefail
repo=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
stock="$work/v66_cp1_cap250.zip"
[[ -s $stock ]]
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/official \
    --bind "$work:/work:rw,$repo/2026-competition:/official:ro,$stock:/stock.zip:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
    --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B friday/cap250_fbl2_package/smoke.py \
    /stock.zip /work/v66_cap250_fbl2_trial.zip /input.npz /external.npz /work/known.json /work/unknown.json --device cuda:0
