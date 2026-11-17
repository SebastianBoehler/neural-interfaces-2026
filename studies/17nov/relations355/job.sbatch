#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-rel355
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${R355_WS:?}"
sha256sum -c source.sha256 >/dev/null
ROOT=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
PYTHON="$ROOT/.venv/bin/python"
if [[ "${R355_MODE:?}" == smoke ]]; then
  for FAMILY in lag envelope; do
    "$PYTHON" code/run.py --root "$ROOT" --out "smoke/$FAMILY" --family "$FAMILY" --smoke
  done
else
  "$PYTHON" code/run.py --root "$ROOT" --out "output/${R355_FAMILY:?}" --family "$R355_FAMILY"
fi
sha256sum -c source.sha256 >/dev/null
