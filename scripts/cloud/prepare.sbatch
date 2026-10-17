#!/usr/bin/env bash
#SBATCH --job-name=ni26-prepare
#SBATCH --partition=cpu-galvani
#SBATCH --account=gehler
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
set -euo pipefail
ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT"
mkdir -p .tools/bin runs/cloud runs/baseline
curl -LsSf https://astral.sh/uv/install.sh -o runs/cloud/uv-install.sh
UV_INSTALL_DIR="$ROOT/.tools/bin" UV_NO_MODIFY_PATH=1 \
  sh runs/cloud/uv-install.sh
export PATH="$ROOT/.tools/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python pip \
  -r requirements/cloud.txt
export PATH="$ROOT/.venv/bin:$PATH"
export BENCHOPT_DATA_HOME="$ROOT/data"
export COMPET_DEVICE=cpu
export BENCHOPT_DEBUG=true
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset NEURALBENCH_EXTRACT_CPUS
python -m pip freeze > runs/cloud/requirements.lock.txt
cd 2026-competition
benchopt prepare tracks/bci_decoding --config tracks/bci_decoding/training.yml
cd "$ROOT"
python scripts/audit_proteus.py
cp runs/baseline/data_audit.json runs/cloud/data_audit.json
touch runs/cloud/PREPARED
