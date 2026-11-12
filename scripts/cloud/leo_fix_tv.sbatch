#!/bin/bash
#SBATCH --job-name=ni26-fix-tv
#SBATCH --account=AIFAC_P02_1675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=runs/diag/leo_fix_tv-%j.out
# torchvision from PyPI does not match the cu126 torch in .venv312 ("operator torchvision::nms does not exist");
# replace it by the cu126 build. Usage: sbatch scripts/cloud/leo_fix_tv.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset PYTHONPATH; export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" UV_CACHE_DIR="$PWD/.uv-cache"
.venv/bin/python -m uv pip install -q -p .venv312/bin/python --reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cu126
.venv312/bin/python -c "import torch, torchvision; from torchvision.ops import nms; print(torch.__version__, torchvision.__version__)"; echo TV_OK
