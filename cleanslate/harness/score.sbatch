#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=cs-score
#SBATCH --output=%x-%j.out
# python score.py RUN inside the worker image (official metric code); env CS_REL, CS_RUN. Optional $1 = score.py extra args.
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --env PYTHONDONTWRITEBYTECODE=1 \
  --bind "$CS_REL/cleanslate:/cs:ro,$CS_RUN:/work:rw,$base/neural-interfaces-2026/2026-competition:/official:ro,$base/neural-interfaces-2026/data/proteus_windows.npz:/data/proteus_windows.npz:ro" \
  --pwd /work "$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif" python -B /cs/harness/score.py /work
