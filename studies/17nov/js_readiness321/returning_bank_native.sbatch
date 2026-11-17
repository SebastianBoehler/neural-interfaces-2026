#!/bin/bash
#SBATCH --job-name=ni26-js321-returning-native-bank
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
base=/mnt/lustre/work/gehler/gfh527
sif="$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif"
test -f "$sif"
mkdir banks
sha256sum "$sif" > worker_image.sha256
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --nv \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
 --bind "$base:$base:ro" --bind "$PWD:/probe:ro" --bind "$PWD/banks:/probe/banks:rw" \
 --pwd /probe "$sif" python /probe/returning_bank_native.py P04_s23 0
