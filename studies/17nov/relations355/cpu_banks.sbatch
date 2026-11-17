#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-rel355-CPU-B1
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${R355_CPU:?}"
sha256sum -c instrument.sha256 >/dev/null
(cd "$R355_WS" && sha256sum -c source.sha256 >/dev/null)
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python instrument/cpu_coordinate.py
