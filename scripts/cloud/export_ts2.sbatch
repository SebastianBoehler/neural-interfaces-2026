#!/usr/bin/env bash
#SBATCH --job-name=ni26-rts2-export
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,cpu-galvani
#SBATCH --exclude=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
# Usage: sbatch --export=ALL,NI26_SOURCE_REV=<commit> scripts/cloud/export_ts2.sbatch SRC_DIR OUT_DIR --src-harness H [args]
# Trains the Hilbert TS core on all public windows and builds/checks OUT_DIR = SRC_DIR + that core as a second
# tangent member (scripts/export_ts2.py; docs/AUDIT_SBN.md). CPU only. Nothing is uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data" PYTHONWARNINGS=ignore NI26_ROOT="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/export_ts2.py "$@"
