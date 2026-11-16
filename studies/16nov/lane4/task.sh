#!/bin/bash
set -euo pipefail
export NI26_WS="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
sha256sum -c studies/16nov/lane4/inputs.sha256
exec .venv/bin/python -I studies/16nov/lane4/run.py "$@"
