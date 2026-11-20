#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
case "${1:?cluster required}" in
  leonardo)
    module load python/3.11.7
    unset PYTHONPATH
    source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate
    ;;
  galvani)
    export PATH=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin:$PATH
    ;;
  *) exit 2 ;;
esac
shift
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CS_DATA="$PWD/data/proteus_windows.npz"
export CS_OFFICIAL="$PWD/2026-competition/tracks/bci_decoding"
script=${1:?script required}
shift
case "$script" in
  base|fold)
    root=${1:?run directory required}
    shift
    exec python -u "cleanslate/m1/$script.py" "$root" "${SLURM_ARRAY_TASK_ID:?}" "$@"
    ;;
  external)
    exec python -u cleanslate/m1/external.py "${SLURM_ARRAY_TASK_ID:?}" "$@"
    ;;
  report)
    exec python -u "cleanslate/m1/$script.py" "$@"
    ;;
  *) exit 2 ;;
esac
