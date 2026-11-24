#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-niv-enrol
#SBATCH --output=logs/niv-enrol-%A.out
# Usage (release dir): sbatch enrol.sbatch SRC_ZIP_NAME TAG   -> zips/enrolled/TAG_k0..k8.zip
set -euo pipefail
zipname=${1:?}; tag=${2:?}
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}
cd "$work"; mkdir -p zips/enrolled
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=4,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work \
  --bind "$work:/work:rw,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$work/scherer_rows.tsv.gz:/rows.tsv.gz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B niv/build_enrolled.py zips/$zipname zips/enrolled "$tag"
