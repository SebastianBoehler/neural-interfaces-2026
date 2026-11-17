#!/bin/bash
#SBATCH --job-name=ni26-js321-profile
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:06:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sif=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/runs/cloud/ni26_worker.sif
test -f "$sif"
expected=$(cut -d' ' -f1 /mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_worker2/output_cuda/worker_image.sha256)
actual=$(sha256sum "$sif" | cut -d' ' -f1)
test "$actual" = "$expected"
out="$PWD/output_profile"
mkdir "$out"
sha256sum "$sif" > "$out/worker_image.sha256"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --nv \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
 --bind "$PWD:/probe:ro" --bind "$out:/out:rw" \
 --bind /mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_orchestration4:/experiment:ro \
 --bind /mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_active5:/active:ro \
 --bind /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/2026-competition:/official:ro \
 --bind /mnt/lustre/work/gehler/gfh527/ni26_s304_coral_prod02_20261116/data/proteus_windows.npz:/input.npz:ro \
 --pwd /probe "$sif" python /probe/profile_bank.py
