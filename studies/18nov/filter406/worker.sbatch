#!/bin/bash
#SBATCH -J ni26-filter406
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 00:20:00
set -eu
base=/mnt/lustre/work/gehler/gfh527
cd "$SLURM_SUBMIT_DIR"
sha256sum -c source.sha256
owner="$base/ni26_js321_20261117_returning_pair_cpu1"
(cd "$owner" && sha256sum -c source.sha256 >/dev/null)
out="$PWD/cell_${SLURM_ARRAY_TASK_ID:?}"
mkdir "$out"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
 --bind "$base:$base:ro" --bind "$out:/out:rw" --pwd "$owner" \
 "$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif" python "$PWD/run.py" "$SLURM_ARRAY_TASK_ID"
