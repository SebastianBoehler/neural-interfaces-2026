#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=cs-refcap
#SBATCH --output=%x-%j.out
# env CS_REL, CS_RUN (RUN dir to create, e.g. $CS_REL/runs/cap250/c25d_full0). Reads the c25d full0 bank, then scores.
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527; bank=$base/ni26_friday_c25d_held_v1_20261119/full0.npz
mkdir -p "$CS_RUN"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --env PYTHONDONTWRITEBYTECODE=1 \
  --bind "$CS_REL/cleanslate:/cs:ro,$CS_RUN:/work:rw,$bank:/bank.npz:ro,$base/neural-interfaces-2026/2026-competition:/official:ro,$base/neural-interfaces-2026/data/proteus_windows.npz:/data/proteus_windows.npz:ro" \
  --pwd /work "$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif" bash -c "python -B /cs/harness/reference_cap250.py /bank.npz /work && python -B /cs/harness/score.py /work"
