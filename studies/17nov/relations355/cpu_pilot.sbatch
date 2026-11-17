#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:02:00
#SBATCH --job-name=ni26-rel355-cpu-source
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${R355_CPU:?}"
sha256sum -c source.sha256 >/dev/null
(cd "$R355_WS" && sha256sum -c source.sha256 >/dev/null)
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python cpu_pilot.py
