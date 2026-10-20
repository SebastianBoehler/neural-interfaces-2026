#!/usr/bin/env bash
#SBATCH --job-name=ni26-sealed
#SBATCH --partition=a100-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
# Usage: sbatch [--qos=vs] --export=ALL,NI26_SOURCE_REV=$(git rev-parse HEAD) \
#            scripts/cloud/build_sealed.sbatch scripts/sealed_r6nf.json [OUT_DIR]
# Rebuilds a sealed package from scratch (nets trained on the A100), checks it
# against the harness and the config's reference package, then runs the
# official ingestion + scoring replay with COMPET_DEVICE=cuda. Nothing is
# uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f runs/cloud/PREPARED
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset NEURALBENCH_EXTRACT_CPUS
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
OUT="${2:-runs/sealed/$(basename "$1" .json)_a100_$SLURM_JOB_ID}"
python scripts/build_sealed.py "$1" "$OUT" --device cuda
python scripts/replay_package.py "$OUT" "${OUT}_replay_a100" cuda
