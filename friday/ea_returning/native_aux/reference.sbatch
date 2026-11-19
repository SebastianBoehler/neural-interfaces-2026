#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-return-core2-reference
#SBATCH --output=core2-ref-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
recipe=$base/ni26_js321_20261117_returning2/source
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c ref_source.sha256
sha256sum --quiet -c fit_inputs.sha256
sha256sum --quiet -c reference_inputs.sha256
export NI26_ROOT="$recipe"
export PYTHONPATH="$recipe/experiments:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$repo/.venv/bin/python" -B hilbert_reference.py
