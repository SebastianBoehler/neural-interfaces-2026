#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-rel355-B1banks
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${R355_B1:?}"
sha256sum -c instrument.sha256 >/dev/null
(cd "$R355_WS" && sha256sum -c source.sha256 >/dev/null)
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python instrument/b1_banks.py
