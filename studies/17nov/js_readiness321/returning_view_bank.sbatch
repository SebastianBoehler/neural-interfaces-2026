#!/bin/bash
#SBATCH --job-name=ni26-js321-full-raw256
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:25:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
base=/mnt/lustre/work/gehler/gfh527
sif="$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif"
folds=(P04_s23 P10_s23)
fold=${folds[${SLURM_ARRAY_TASK_ID:?}]}
out="$PWD/$fold"
mkdir "$out"
cp source.sha256 "$out/source.sha256"
sha256sum "$sif" > "$out/worker_image.sha256"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl \
 --bind "$base:$base:ro" --bind "$PWD:/probe:ro" --bind "$out:/out:rw" \
 --pwd /out "$sif" python /probe/returning_view_bank.py "$fold"
