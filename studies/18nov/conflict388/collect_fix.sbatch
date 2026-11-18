#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-galvani,2080-preemptable-galvani,a100-galvani,a100-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-score388
set -euo pipefail
here=${SLURM_SUBMIT_DIR:?}
cd "$here"
sha256sum -c source.sha256
export NI26_DEVICE=cpu OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/lustre/work/gehler/gfh527/ni26_conflict388_v1_20261118
cd "$PYTHONPATH"
sha256sum -c source.sha256
grep -Fx OK replay.log
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B "$here/score388.py"
