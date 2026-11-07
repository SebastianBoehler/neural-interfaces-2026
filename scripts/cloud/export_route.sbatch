#!/usr/bin/env bash
#SBATCH --job-name=ni26-route-export
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,cpu-galvani
#SBATCH --exclude=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
# Usage: sbatch --export=ALL,NI26_SOURCE_REV=<commit> scripts/cloud/export_route.sbatch SRC_DIR OUT_DIR --route R --hi HI --src-harness H --hi-harness H2 [args]
# Builds/checks OUT_DIR = SRC_DIR (cand_rec_ts2_tsem) + the returning-person routing stage after TSEM
# (scripts/export_route.py; docs/AUDIT_SBN.md). CPU only. Nothing is uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" BENCHOPT_DATA_HOME="$PWD/data" PYTHONWARNINGS=ignore NI26_ROOT="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/export_route.py "$@"
