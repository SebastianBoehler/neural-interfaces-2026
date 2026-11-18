#!/bin/bash
#SBATCH --account=gehler
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-sinc-orders
#SBATCH --output=sinc-orders-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
HERE=${SLURM_SUBMIT_DIR:?}
cd "$HERE"
sha256sum --quiet -c source.sha256
(
    cd "$REPO"
    sha256sum --quiet -c "$HERE/fresh_sinc_inputs.sha256"
    sha256sum --quiet -c "$HERE/refit_inputs.sha256"
)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
case "${1:?expected replay or collect}" in
    replay)
        printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID + 1))"
        exec "$REPO/.venv/bin/python" -B fresh_sinc_orders.py "$person"
        ;;
    collect) exec "$REPO/.venv/bin/python" -B fresh_sinc_orders.py collect ;;
    *) exit 2 ;;
esac
