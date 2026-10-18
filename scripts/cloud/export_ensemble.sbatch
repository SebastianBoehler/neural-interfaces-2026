#!/usr/bin/env bash
#SBATCH --job-name=ni26-export
#SBATCH --partition=a100-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
# Usage: sbatch scripts/cloud/export_ensemble.sbatch official|all [rb]
# "rb" adds causal class rebalancing (alpha 0.5, k 5) and exports to
# runs/submissions/ensemble_<train>_rb/.
# Trains the v2 + 5x aligned + 5x raw ShallowConvNet ensemble (v2 weight 1/3)
# and exports runs/submissions/ensemble_<train>/ with the agreement check,
# then replays the package through the official ingestion + scoring
# programs on this A100 node. Nothing is uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f runs/cloud/PREPARED
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
NAME="ensemble_$1"; RB=()
if [ "${2:-}" = rb ]; then NAME="${NAME}_rb"; RB=(--rb-alpha 0.5 --rb-k 5); fi
python scripts/train_ensemble.py --train "$1" --deep both --seeds 5 \
  --v2-weight 0.3333333333333333 "${RB[@]}" --out "runs/submissions/$NAME"
rm -rf "runs/submissions/${NAME}_replay_a100"
python scripts/replay_package.py "runs/submissions/$NAME" \
  "runs/submissions/${NAME}_replay_a100" cuda
