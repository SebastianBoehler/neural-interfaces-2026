#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-ear-bank
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
parent=$base/ni26_friday_ea_matrix_20261118
revision=c625ecc736a881836a25f85a741ea9147bf698bf
sha256sum --quiet -c source.sha256
sha256sum --quiet -c inputs.sha256
sha256sum --quiet -c smoke.sha256
(cd "$parent" && sha256sum --quiet -c source.sha256)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$parent:$parent/$revision/experiments:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
"$repo/.venv/bin/python" -B returning_collect.py "$repo"
