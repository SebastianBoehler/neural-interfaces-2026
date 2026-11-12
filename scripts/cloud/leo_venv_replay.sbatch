#!/bin/bash
#SBATCH --job-name=ni26-venv-replay
#SBATCH --account=AIFAC_P02_1675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=runs/diag/leo_venv_replay-%j.out
# Adds the official ingestion stack (requirements/cloud.txt pins) to $WORK/ni26/.venv on a serial node, keeping the
# cu126 torch. Usage: sbatch scripts/cloud/leo_venv_replay.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH
echo "torch==2.14.1+cu126" > /tmp/ni26_constraints.txt
.venv/bin/python -m pip install -q --no-cache-dir -r requirements/cloud.txt -c /tmp/ni26_constraints.txt --extra-index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -q --no-cache-dir --force-reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import torch, benchopt, neuralset, braindecode; print(torch.__version__, benchopt.__version__, neuralset.__version__)"; echo VENV_REPLAY_OK
