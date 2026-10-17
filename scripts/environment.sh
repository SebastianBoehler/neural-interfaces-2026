#!/usr/bin/env bash
# Source from Bash in any directory; artifacts stay in this workspace.
PROTEUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$PROTEUS_ROOT/.venv/bin:$PATH"
export BENCHOPT_DATA_HOME="$PROTEUS_ROOT/data"
export COMPET_DEVICE=cpu
unset NEURALBENCH_EXTRACT_CPUS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BENCHOPT_DEBUG=true
