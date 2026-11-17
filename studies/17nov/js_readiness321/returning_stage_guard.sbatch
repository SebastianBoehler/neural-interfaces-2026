#!/bin/bash
#SBATCH --job-name=ni26-js321-fold-stage-guard
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
base=/mnt/lustre/work/gehler/gfh527
sif="$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif"
mkdir output
sha256sum "$sif" > output/worker_image.sha256
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
 --bind "$base:$base:ro" --bind "$PWD:/probe:ro" --bind "$PWD/output:/out:rw" \
 --bind "$base/neural-interfaces-2026/2026-competition:/official:ro" \
 --bind "$base/ni26_js321_20261117_active5:/active:ro" \
 --bind "$base/ni26_js321_20261117_inputreuse2:/experiment:ro" \
 --bind "$base/ni26_js321_20261117_returning_context_pilot2:/context:ro" \
 --pwd /probe "$sif" python /probe/returning_stage_guard.py
