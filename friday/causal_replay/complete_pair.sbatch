#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:12:00
#SBATCH --job-name=ni26-causal-full-pair
#SBATCH --output=paired-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REBUILD=$BASE/ni26_friday_causal_rebuild_20261118
HERE=${SLURM_SUBMIT_DIR:?}
cd "$HERE"
sha256sum --quiet -c source.sha256
(
    cd "$BASE/ni26_s318_coral_fbl_20261117"
    sha256sum --quiet -c "$HERE/owner.sha256"
    sha256sum --quiet -c "$HERE/terminal_inputs.sha256"
)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=$BASE/neural-interfaces-2026/.venv/bin/python
case "${1:?expected predict or score}" in
    predict)
        exec "$PY" -B complete_pair.py \
            "$REBUILD/terminal_full0_v2/zero.json" 9b9dc4070b234a6df2c2d806c7ad5f2ddf2dbd231370fae688369a78b046663c \
            "$REBUILD/terminal_full0_v3/active_prefix.json" d9b53c4f4289de0f784c6754d21b9018e805045668ad9a8e52fbc142d85fb490 \
            "$REBUILD/refit_full0_matrix_v2" 841d35a4e3d3bcd9dba44e4f7b03ef896622d64fa922b5da9ad7afb367cc698c
        ;;
    score)
        read -r proof _ < <(sha256sum matched_full0.json)
        exec "$PY" -B complete_score.py "$proof"
        ;;
    *) exit 2 ;;
esac
