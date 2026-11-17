#!/bin/bash
#SBATCH --job-name=ni26-js321-conditional-pair
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
running=$(squeue --me -h -r -t RUNNING | wc -l)
active=$(squeue --me -h -r | wc -l)
if ((running>32 || active>244)); then echo "CAPACITY_GUARD_FAILED running=$running active=$active" >&2; exit 1; fi
base=/mnt/lustre/work/gehler/gfh527
sif="$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif"
out="$PWD/cell_${SLURM_ARRAY_TASK_ID:?}"
mkdir "$out"
sha256sum "$sif" > "$out/worker_image.sha256"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --nv \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 --bind "$base:$base:ro" --bind "$PWD:/probe:ro" --bind "$out:/out:rw" \
 --pwd /probe "$sif" python /probe/returning_pair_cell.py "$SLURM_ARRAY_TASK_ID"
