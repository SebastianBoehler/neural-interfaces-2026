#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s339-release_chain
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_calibration339_setup_v1_20261117
sha256sum -c source.sha256 >/dev/null
export ASSEMBLY_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_calibration339_setup_v1_20261117 NI26_WS=/mnt/lustre/work/gehler/gfh527/ni26_calibration339_chain_v1_20261117 NI26_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B /mnt/lustre/work/gehler/gfh527/ni26_calibration339_chain_v1_20261117/studies/17nov/representation_chain/release339.py
