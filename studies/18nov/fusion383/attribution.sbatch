#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-galvani,2080-preemptable-galvani,a100-galvani,a100-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-383attr
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256
grep -Fx OK /mnt/lustre/work/gehler/gfh527/ni26_fusion383_v1_20261118/replay.log
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B attribution.py "$1"
