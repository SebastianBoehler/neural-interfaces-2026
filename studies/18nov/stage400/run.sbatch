#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,v100-galvani
#SBATCH -c 1
#SBATCH --mem=5G
#SBATCH -t 00:05:00
#SBATCH -J ni26-st400
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
arms=(control RB L M D)
id=${SLURM_ARRAY_TASK_ID:-0}
case "$1" in
 capture) "$py" capture.py "${arms[id/8]}" "${orders[id%8]}" ;;
 count) "$py" count.py "${orders[id]}" ;;
 score) "$py" score.py ;;
 replay) (cd /mnt/lustre/work/gehler/gfh527/ni26_policy395_chain_v1_20261118/chain && "$py" replay/check.py) > replay.log; grep -Fx OK replay.log ;;
esac
