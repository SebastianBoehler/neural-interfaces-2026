#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-lag364-chain
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c chain-source.sha256
(cd chain && sha256sum -c ../runtime.sha256 >/dev/null)
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r arm order < <("$PY" -c 'import json,os;print(*json.load(open("cells.json"))[int(os.environ["SLURM_ARRAY_TASK_ID"])])')
else
    arm=${1:?}; order=${2:?}
fi
"$PY" -B chain.py "$arm" "$order"
