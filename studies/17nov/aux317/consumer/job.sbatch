#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s315-chain
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c representation-source.sha256 >/dev/null
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
DIR=studies/17nov/representation_chain
case ${1:?} in
 control)
  orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
  "$PY" -B "$DIR/cell.py" control "${orders[${SLURM_ARRAY_TASK_ID:?}]}" ;;
 smoke)
  "$PY" -B "$DIR/cell.py" zero_repr within0
  arm=$("$PY" -c 'import json;print(json.load(open("representation_config.json"))["arms"][0])')
  "$PY" -B "$DIR/cell.py" "$arm" within0
  "$PY" -B "$DIR/release.py" ;;
 remaining)
  read -r arm order < <("$PY" -c 'import json,os;print(*json.load(open("remaining_cells.json"))[int(os.environ["SLURM_ARRAY_TASK_ID"])])')
  "$PY" -B "$DIR/cell.py" "$arm" "$order" ;;
 *) exit 2 ;;
esac
