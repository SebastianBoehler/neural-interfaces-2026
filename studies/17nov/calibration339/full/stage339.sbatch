#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s339-stage339
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_calibration339_chain_v1_20261117
sha256sum -c representation-source.sha256 >/dev/null
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B studies/17nov/representation_chain/stage339.py
