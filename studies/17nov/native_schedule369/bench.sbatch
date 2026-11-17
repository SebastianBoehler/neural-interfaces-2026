#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-schedule369-a100
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
base=/mnt/lustre/work/gehler/gfh527
repo="$base/neural-interfaces-2026"
qualified="$base/ni26_native366_exact_ea_v1_20261117"
sha256sum -c source.sha256
(cd "$qualified" && sha256sum --quiet -c stress.sha256 && sha256sum --quiet -c inputs.sha256)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$repo/.venv/bin/python" variants.py "$qualified/stress_native" "$PWD"
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs   --env OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1   --bind "$PWD:/work:rw,$qualified/stress_native:/stock:ro,$repo/2026-competition:/official:ro,$repo/data/proteus_windows.npz:/input.npz:ro"   --pwd /work "$repo/runs/cloud/ni26_worker.sif" python bench.py
sha256sum -c source.sha256
