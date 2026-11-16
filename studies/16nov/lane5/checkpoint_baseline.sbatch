#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/checkpoint-%j.out
#SBATCH --error=logs/checkpoint-%j.err
set -euo pipefail
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LANE5_LIBRARY="$PWD/liblane5.so"
sha256sum -c source.sha256
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I test_publication.py
