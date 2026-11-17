#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:06:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s339-job339
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_calibration339_chain_v1_20261117
sha256sum -c representation-source.sha256 >/dev/null
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
case ${1:?} in
 control) orders=(within0 within1 within2 within3 full0 full1 runs0 runs1); arm=control; order=${orders[${SLURM_ARRAY_TASK_ID:?}]} ;;
 cell) arm=${2:?}; order=${3:?} ;;
 remaining) read -r arm order < <("$PY" -c 'import json,os;print(*json.load(open("remaining_cells.json"))[int(os.environ["SLURM_ARRAY_TASK_ID"])])') ;;
 *) exit 2 ;;
esac
"$PY" -B studies/17nov/representation_chain/cell339.py "$arm" "$order"
