#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-return-core2-fit
#SBATCH --output=core2-fit-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
recipe=$base/ni26_js321_20261117_returning2/source
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c fit_source.sha256
sha256sum --quiet -c fit_inputs.sha256
export NI26_ROOT="$recipe"
export PYTHONPATH="$recipe/experiments:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$repo/.venv/bin/python" -B fit_hilbert.py
