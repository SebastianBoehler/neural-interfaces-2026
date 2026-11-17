#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:07:00
#SBATCH --job-name=ni26-chain376c
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD/chain" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
if [[ "$1" == setup ]]; then
    "$py" -B context.py
    (cd chain && "$py" replay/check.py) > replay.log
    grep -Fx OK replay.log
    exit 0
fi
(cd chain && sha256sum -c ../runtime.sha256 >/dev/null)
grep -Fx OK replay.log
case "$1" in
    parent)
        (cd /mnt/lustre/work/gehler/gfh527/ni26_consumer376_v1_20261117/chain && sha256sum -c ../runtime.sha256 >/dev/null)
        (cd chain && "$py" -B ../verify.py parent) ;;
    stage) (cd chain && "$py" -B ../verify.py stage) ;;
    score) (cd chain && "$py" -B ../score.py) ;;
    cell)
        arms=(zero_repr A G control)
        orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
        id=${SLURM_ARRAY_TASK_ID:?}
        cd chain
        "$py" -B studies/17nov/representation_chain/cell.py "${arms[id/8]}" "${orders[id%8]}" ;;
    *) exit 2 ;;
esac
