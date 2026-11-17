#!/bin/bash
#SBATCH --job-name=ni26-policy350-whole
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:09:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
running=$(squeue --me -h -r -t RUNNING | wc -l)
active=$(squeue --me -h -r | wc -l)
if ((running>32 || active>244)); then exit 1; fi
base=/mnt/lustre/work/gehler/gfh527
out="$PWD/cell_${SLURM_ARRAY_TASK_ID:?}"
mkdir "$out"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
 --bind "$base:$base:ro" --bind "$PWD:/probe:ro" --bind "$out:/out:rw" \
 --pwd /probe "$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif" python cell.py "$SLURM_ARRAY_TASK_ID"
