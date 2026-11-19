#!/bin/bash
#SBATCH --account=gehler
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:07:00
#SBATCH --job-name=ni26-within0-pair
#SBATCH --output=pair-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
HERE=${SLURM_SUBMIT_DIR:?}
cd "$HERE"
sha256sum --quiet -c source.sha256
(cd "$BASE" && sha256sum --quiet -c "$HERE/within0_warm_inputs.sha256")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/experiments:$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
family=${1:?family}
case "${3:-within0}" in
    within0) entry=within0_matched_fit.py; release=within0_shared_bridge_v1 ;;
    runs0) entry=runs0_pair.py; release=runs0_shared_bridge_v1 ;;
    *) exit 2 ;;
esac
bridge=$(sha256sum "$HERE/../$release/status.json" | cut -d ' ' -f 1)
if [[ "$family" == sinc ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
python="$REPO/.venv/bin/python"
case "${2:?mode}" in
    smoke) exec "$python" -B "$entry" "$family" "$bridge" --smoke ;;
    production)
        proof=$("$python" -B "$entry" "$family" "$bridge" --verify-smoke)
        export NI26_WITHIN0_PAIR_SMOKE="$proof"
        exec "$python" -B "$entry" "$family" "$bridge"
        ;;
    *) exit 2 ;;
esac
