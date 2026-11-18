#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-native-inner-collect
#SBATCH --output=inner-collect-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c source.sha256
sha256sum --quiet -c collect_inputs.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$base/neural-interfaces-2026/.venv/bin/python" -B collect.py
