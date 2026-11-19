#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-within0-sinc
#SBATCH --output=sinc-bridge-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
HERE=${SLURM_SUBMIT_DIR:?}
cd "$HERE"
sha256sum --quiet -c source.sha256
(
    cd "$BASE/ni26_s318_coral_fbl_20261117"
    sha256sum --quiet -c "$HERE/owner.sha256"
    sha256sum --quiet -c "$HERE/recognition.sha256"
)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python="$BASE/neural-interfaces-2026/.venv/bin/python"
case "${1:-within0}" in
    within0) exec "$python" -B within0_sinc_bridge.py ;;
    runs0) exec "$python" -B runs0_bridge.py ;;
    *) exit 2 ;;
esac
