#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --job-name=ni26-within0-warm
#SBATCH --output=warm-%A_%a.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
HERE=${SLURM_SUBMIT_DIR:?}
cd "$HERE"
sha256sum --quiet -c source.sha256
(cd "$BASE" && sha256sum --quiet -c "$HERE/within0_warm_inputs.sha256")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/experiments:$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
python="$REPO/.venv/bin/python"
case "${1:?expected smoke, remaining or collect}" in
    smoke) exec "$python" -B within0_warm.py "${SLURM_ARRAY_TASK_ID:?}" --smoke ;;
    remaining)
        proof=$("$python" -B within0_warm_collect.py "${SLURM_ARRAY_TASK_ID:?}")
        export NI26_WITHIN0_WARM_SMOKE_SHA="$proof"
        exec "$python" -B within0_warm.py "${SLURM_ARRAY_TASK_ID:?}"
        ;;
    collect) exec "$python" -B within0_warm_collect.py ;;
    *) exit 2 ;;
esac
