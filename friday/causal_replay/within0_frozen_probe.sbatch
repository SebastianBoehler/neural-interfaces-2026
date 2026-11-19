#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-within0-frozen
#SBATCH --output=within0-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
HERE=${SLURM_SUBMIT_DIR:?}
cd "$HERE"
sha256sum --quiet -c source.sha256
(cd "$BASE" && sha256sum --quiet -c "$HERE/within0_frozen_inputs.sha256")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/experiments:$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
case "${1:?expected fold or collect}" in
    fold)
        proof=$("$REPO/.venv/bin/python" -B within0_frozen_collect.py p04)
        export NI26_WITHIN0_P04_SHA="$proof"
        exec "$REPO/.venv/bin/python" -B within0_frozen_probe.py "${SLURM_ARRAY_TASK_ID:?}"
        ;;
    collect) exec "$REPO/.venv/bin/python" -B within0_frozen_collect.py collect ;;
    *) exit 2 ;;
esac
