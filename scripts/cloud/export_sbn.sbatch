#!/usr/bin/env bash
#SBATCH --job-name=ni26-sbn-export
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --exclude=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
# Usage: sbatch --export=ALL,NI26_SOURCE_REV=<commit> scripts/cloud/export_sbn.sbatch SRC_DIR OUT_DIR [args]
# Trains fb_sbn_rot05 seeds 0-5 on all public windows (cached in runs/sbn), exports the SJit members and
# builds/checks OUT_DIR from SRC_DIR (scripts/export_sbn.py; docs/AUDIT_SBN.md). Nothing is uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data" PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/check_sjit.py --cuda
python scripts/export_sbn.py "$@"
