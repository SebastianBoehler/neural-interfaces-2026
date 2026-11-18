#!/bin/bash
#SBATCH --account=gehler
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-fresh-pair-refit
#SBATCH --output=pair-refit-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
HERE=${SLURM_SUBMIT_DIR:?}
cd "$HERE"
sha256sum --quiet -c source.sha256
(
    cd "$REPO"
    sha256sum --quiet -c "$HERE/refit_inputs.sha256"
    sha256sum --quiet -c "$HERE/fresh_sinc_inputs.sha256"
)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if [[ "${1:?member family required}" == sinc ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
exec "$REPO/.venv/bin/python" -B restore_refit_pair.py "$@"
