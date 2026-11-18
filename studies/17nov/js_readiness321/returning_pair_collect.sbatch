#!/bin/bash
#SBATCH --job-name=ni26-js321-conditional-collect
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:04:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python returning_pair_collect.py preflight > collect_preflight.log
work="$PWD"
cd /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python replay/check.py > "$work/replay_check.txt"
cd "$work"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python returning_pair_collect.py collect
