#!/bin/bash
#SBATCH --account=gehler
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-laneE-time
#SBATCH --output=logs/time-%x-%j.out
# Usage (Lane E work dir): sbatch -p a100-galvani --gres=gpu:a100:1 timing.sbatch TAGDIR   or  -p 2080-galvani --gres=gpu:rtx2080ti:1
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}
cd "$work"; mkdir -p out
sha256sum --quiet -c zips.sha256
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work/tmp \
  --bind "$work:/work:rw,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B code/lane_e/timing.py /work/out/timing_${gpu}.json cuda:0 \
  zips/v66_cp1_cap250.zip:cap250 zips/E0.zip:E0 zips/Ea.zip:Ea zips/Eb.zip:Eb zips/Ec.zip:Ec zips/Ed.zip:Ed zips/Ed_g2.zip:Ed_g2
