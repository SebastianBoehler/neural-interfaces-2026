#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:06:00
#SBATCH --job-name=ni26-within0-complete
#SBATCH --output=within0-complete-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
HERE=${SLURM_SUBMIT_DIR:?}
cd "$HERE"
sha256sum --quiet -c source.sha256
(
    cd "$BASE/ni26_s318_coral_fbl_20261117"
    sha256sum --quiet -c "$HERE/owner.sha256"
    sha256sum --quiet -c "$HERE/recognition.sha256"
    sha256sum --quiet -c "$HERE/complete_terminal_inputs.sha256"
)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python="$BASE/neural-interfaces-2026/.venv/bin/python"
case "${1:?terminal mode}" in
    runs0-zero) exec "$python" -B runs0_zero.py ;;
    runs0-predict|runs0-score) exec "$python" -B runs0_terminal.py "${1#runs0-}" "${@:2}" ;;
    full0-predict|full0-score) exec "$python" -B full0_common_terminal.py "${1#full0-}" "${@:2}" ;;
    *) exec "$python" -B within0_complete.py "$@" ;;
esac
