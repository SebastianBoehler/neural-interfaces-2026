#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=3G
#SBATCH -t 5
#SBATCH -J ni26-score459
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$b/neural-interfaces-2026/experiments
sha256sum -c --status collector.sha256
(cd "$b" && sha256sum -c --status "$SLURM_SUBMIT_DIR/external_parents.sha256")
"$b/neural-interfaces-2026/.venv/bin/python" external_score.py "$SLURM_SUBMIT_DIR"
