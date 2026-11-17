#!/bin/bash
#SBATCH --job-name=ni26-js321-returning-native-bank
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
base=/mnt/lustre/work/gehler/gfh527
sif="$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif"
test -f "$sif"
index=${SLURM_ARRAY_TASK_ID:?}
if ((index<5)); then fold=P04_s23; seed=$((index+1)); else fold=P10_s23; seed=$((index-5)); fi
out="$PWD/banks_${fold}_s${seed}"
mkdir "$out"
sha256sum "$sif" > "$out/worker_image.sha256"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --nv \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
 --bind "$base:$base:ro" --bind "$PWD:/probe:ro" --bind "$out:/probe/banks:rw" \
 --pwd /probe "$sif" python /probe/returning_bank_native.py "$fold" "$seed"
