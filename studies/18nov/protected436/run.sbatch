#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:12:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c source.sha256
(( $(squeue --me -h -r -t RUNNING | wc -l) <= 32 ))
base=/mnt/lustre/work/gehler/gfh527
cell=${SLURM_ARRAY_TASK_ID:-"-1"}
out="$PWD/cell_$cell"
mkdir "$out"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
 --bind "$base:$base:ro" --bind "$PWD:/probe:ro" --bind "$out:/out:rw" \
 --pwd /probe "$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif" python cell.py "$cell"
