#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-common-fbl
#SBATCH --output=common-fbl-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
here=${SLURM_SUBMIT_DIR:?}
cd "$here"
sha256sum --quiet -c source.sha256
(cd "$base" && sha256sum --quiet -c "$here/within0_frozen_inputs.sha256")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/experiments:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
python="$repo/.venv/bin/python"
case "${1:?smoke, fold or collect}" in
    smoke) exec "$python" -B within0_fresh_fbl.py fold 1 ;;
    fold)
        "$python" -B within0_fresh_fbl.py check
        exec "$python" -B within0_fresh_fbl.py fold "${SLURM_ARRAY_TASK_ID:?}"
        ;;
    collect) exec "$python" -B within0_fresh_fbl.py collect ;;
    *) exit 2 ;;
esac
