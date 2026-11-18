#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-galvani,2080-preemptable-galvani,a100-galvani,a100-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=00:07:00
#SBATCH --job-name=ni26-cp395
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD/chain" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
case "$1" in
 setup)
    "$py" -B setup_chain.py
    (cd chain && "$py" replay/check.py) > replay.log
    grep -Fx OK replay.log
    (cd chain && "$py" -B ../finish395.py stage) ;;
 cell)
    (cd chain && sha256sum -c ../runtime.sha256 >/dev/null)
    id=${SLURM_ARRAY_TASK_ID:-${2:?}}
    arms=(control zero_repr RB L M D)
    orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
    cd chain
    "$py" -B studies/17nov/representation_chain/cell.py "${arms[id/8]}" "${orders[id%8]}" ;;
 score)
    (cd chain && sha256sum -c ../runtime.sha256 >/dev/null)
    grep -Fx OK replay.log
    cd chain
    "$py" -B ../finish395.py score ;;
esac
