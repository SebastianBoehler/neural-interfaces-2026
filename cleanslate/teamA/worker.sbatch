#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:15:00
# Slurm-only runtime qualification of an exact packaged Solver in the worker image.
set -euo pipefail
root=$1
pkg=$2
out=$3
base=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=2,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,HF_HUB_OFFLINE=1,PYTHONPATH=/official \
  --bind "$root/cleanslate/teamA:/source:ro,$pkg:/package:ro,$out:/work:rw,$base/2026-competition:/official:ro,$base/data/proteus_windows.npz:/wave.npz:ro" \
  --pwd /work "$base/runs/cloud/ni26_worker.sif" \
  python -B -u /source/runtime_check.py --pkg /package --data /wave.npz --out /work/runtime.json
