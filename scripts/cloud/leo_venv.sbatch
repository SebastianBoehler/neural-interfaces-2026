#!/bin/bash
#SBATCH --job-name=ni26-venv
#SBATCH --account=AIFAC_P02_1675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=runs/diag/leo_venv-%j.out
# Builds $WORK/ni26/.venv on a serial node (never on the login node). Usage: sbatch scripts/cloud/leo_venv.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH
curl -sI https://pypi.org/simple/ | head -1
[ -x .venv/bin/python ] || { python3 -m venv .venv; .venv/bin/python -m pip install -q --upgrade pip; .venv/bin/python -m pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126; }
.venv/bin/python -m pip install -q --no-cache-dir braindecode==1.8.1 mne==1.13.2 scikit-learn scipy numpy joblib
# braindecode pulls a PyPI torchaudio built for CUDA 13; replace it by the cu126 build that matches torch
.venv/bin/python -m pip install -q --no-cache-dir --force-reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip list 2>/dev/null | grep -i "^torch"
.venv/bin/python -c "import torch, braindecode, mne, sklearn; print(torch.__version__, braindecode.__version__)"; echo VENV_OK
