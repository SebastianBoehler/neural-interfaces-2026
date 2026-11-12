#!/bin/bash
#SBATCH --job-name=ni26-venv-replay
#SBATCH --account=AIFAC_P02_1675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=runs/diag/leo_venv_replay-%j.out
# Builds $WORK/ni26/.venv312 for the official ingestion stack on a serial node: neuralset 0.4 needs Python >= 3.12,
# Leonardo's module has 3.11, so uv provides a standalone 3.12. torch stays the cu126 build. Usage: sbatch this file
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH
.venv/bin/python -m pip install -q --no-cache-dir uv
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" UV_CACHE_DIR="$PWD/.uv-cache"
.venv/bin/python -m uv venv -q -p 3.12 .venv312
echo "torch==2.14.1+cu126" > .ni26_constraints.txt
.venv/bin/python -m uv pip install -q -p .venv312/bin/python torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m uv pip install -q -p .venv312/bin/python -r requirements/cloud.txt -c .ni26_constraints.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
.venv/bin/python -m uv pip install -q -p .venv312/bin/python --reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu126
.venv312/bin/python -c "import torch, benchopt, neuralset, braindecode; print(torch.__version__, benchopt.__version__, neuralset.__version__)"; echo VENV_REPLAY_OK
