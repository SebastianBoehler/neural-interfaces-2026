#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-causal-terminal
#SBATCH --output=terminal-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
HERE=$BASE/ni26_friday_causal_rebuild_20261118/terminal_full0_v3
cd "$HERE"
sha256sum --quiet -c source.sha256
(
    cd "$BASE/ni26_s318_coral_fbl_20261117"
    sha256sum --quiet -c "$HERE/owner.sha256"
    sha256sum --quiet -c "$HERE/terminal_inputs.sha256"
)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$BASE/neural-interfaces-2026/.venv/bin/python" -B complete_prefix.py
