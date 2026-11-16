#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-l5-worker
#SBATCH --output=logs/worker-%j.out
#SBATCH --error=logs/worker-%j.err
set -euo pipefail
sha256sum -c source.sha256
sif=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/runs/cloud/ni26_worker.sif
data=$(readlink -f data/proteus_windows.npz)
test -f "$sif"
test -f "$data"
test -f runs/build/liblane5.so
out="$PWD/runs/worker-$SLURM_JOB_ID"
mkdir "$out"
sha256sum "$sif" runs/build/liblane5.so > "$out/artifacts.sha256"
# Hide the host Torch/compiler paths. The image must supply its own libraries.
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --nv \
    --bind "$PWD:/lane5:ro" --bind "$out:/out:rw" --bind "$data:/input.npz:ro" \
    --pwd /lane5 "$sif" \
    python /lane5/studies/16nov/lane5/worker_probe.py \
      --library /lane5/runs/build/liblane5.so \
      --data /input.npz --output /out/report.json
